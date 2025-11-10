use std::{
    collections::HashMap,
    fmt,
    time::{Duration, SystemTime},
};

use actix::Recipient;
use actix_web::{
    post,
    web::{self, Json},
    HttpResponse, Responder,
};
use futures::StreamExt;
use rand::seq::IndexedRandom;
use serde::{Deserialize, Serialize};

use crate::{Content, Message, OpenAiError, ProcessMessages, Role};

const DEFAULT_TIMEOUT_SECS: u64 = 30;
const DEFAULT_MAX_CANDIDATES: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct KnowledgeGraphNode {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub properties: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct KnowledgeGraphEdge {
    pub id: String,
    pub source: String,
    pub target: String,
    pub relation: String,
    #[serde(default)]
    pub properties: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct KnowledgeGraph {
    #[serde(default)]
    pub nodes: Vec<KnowledgeGraphNode>,
    #[serde(default)]
    pub edges: Vec<KnowledgeGraphEdge>,
}

impl KnowledgeGraph {
    fn next_node_id(&self) -> String {
        format!("node-{}", self.nodes.len() + 1)
    }

    fn next_edge_id(&self) -> String {
        format!("edge-{}", self.edges.len() + 1)
    }

    fn find_node_by_label(&self, label: &str) -> Option<&KnowledgeGraphNode> {
        self.nodes
            .iter()
            .find(|n| n.label.eq_ignore_ascii_case(label))
    }

    fn ensure_node(
        &mut self,
        label: &str,
        metadata: &HashMap<String, String>,
    ) -> (KnowledgeGraphNode, bool) {
        if let Some(existing) = self.find_node_by_label(label) {
            return (existing.clone(), false);
        }

        let mut properties = HashMap::new();
        properties.extend(metadata.clone());
        properties
            .entry("label".to_string())
            .or_insert(label.to_string());

        let node = KnowledgeGraphNode {
            id: self.next_node_id(),
            label: label.to_string(),
            properties,
        };
        self.nodes.push(node.clone());
        (node, true)
    }

    fn add_edge(
        &mut self,
        source: &KnowledgeGraphNode,
        target: &KnowledgeGraphNode,
        relation: &str,
        metadata: &HashMap<String, String>,
    ) -> KnowledgeGraphEdge {
        let mut properties = HashMap::new();
        properties.extend(metadata.clone());

        let edge = KnowledgeGraphEdge {
            id: self.next_edge_id(),
            source: source.id.clone(),
            target: target.id.clone(),
            relation: relation.to_string(),
            properties,
        };
        self.edges.push(edge.clone());
        edge
    }

    fn as_prompt_context(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum KarmaVerdict {
    Accept,
    Reject,
    NeedsClarification,
}

impl Default for KarmaVerdict {
    fn default() -> Self {
        Self::NeedsClarification
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema, Default)]
pub struct KarmaCandidateTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    #[serde(default)]
    pub justification: String,
    #[serde(default)]
    pub confidence: f32,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema, Default)]
pub struct KarmaExtractorOutput {
    #[serde(default)]
    pub candidates: Vec<KarmaCandidateTriple>,
    #[serde(default)]
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema, Default)]
pub struct KarmaValidatorDecision {
    pub verdict: KarmaVerdict,
    pub confidence: f32,
    #[serde(default)]
    pub explanation: String,
    pub triple: KarmaCandidateTriple,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema, Default)]
pub struct KarmaPlannerOutput {
    pub objective: String,
    #[serde(default)]
    pub tasks: Vec<String>,
    #[serde(default)]
    pub success_metrics: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct KarmaAgentLog {
    pub agent: KarmaAgentKind,
    pub prompt: String,
    pub response: String,
    pub timestamp_ms: u128,
    pub parsed_successfully: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum KarmaAgentKind {
    Planner,
    Extractor,
    Validator,
}

impl fmt::Display for KarmaAgentKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KarmaAgentKind::Planner => write!(f, "planner"),
            KarmaAgentKind::Extractor => write!(f, "extractor"),
            KarmaAgentKind::Validator => write!(f, "validator"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct KarmaRejectedCandidate {
    pub candidate: KarmaCandidateTriple,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct KarmaEnrichmentRequest {
    pub model: String,
    pub graph: KnowledgeGraph,
    #[serde(default)]
    pub documents: Vec<String>,
    #[serde(default)]
    pub goal: Option<String>,
    #[serde(default)]
    pub instructions: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct KarmaEnrichmentResponse {
    pub plan: Option<KarmaPlannerOutput>,
    pub updated_graph: KnowledgeGraph,
    #[serde(default)]
    pub new_nodes: Vec<KnowledgeGraphNode>,
    #[serde(default)]
    pub new_edges: Vec<KnowledgeGraphEdge>,
    #[serde(default)]
    pub accepted_candidates: Vec<KarmaCandidateTriple>,
    #[serde(default)]
    pub rejected_candidates: Vec<KarmaRejectedCandidate>,
    #[serde(default)]
    pub agent_logs: Vec<KarmaAgentLog>,
}

#[derive(Debug, Clone)]
pub struct KarmaConfig {
    pub request_timeout_secs: u64,
    pub max_candidates_per_document: usize,
}

impl Default for KarmaConfig {
    fn default() -> Self {
        Self {
            request_timeout_secs: DEFAULT_TIMEOUT_SECS,
            max_candidates_per_document: DEFAULT_MAX_CANDIDATES,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KarmaOrchestrator {
    llm_pool: Vec<Recipient<ProcessMessages>>,
    config: KarmaConfig,
}

#[derive(Debug, Clone)]
struct KarmaAgentCall<T> {
    raw: String,
    parsed: Option<T>,
}

impl KarmaOrchestrator {
    pub fn new(llm_pool: Vec<Recipient<ProcessMessages>>, config: Option<KarmaConfig>) -> Self {
        Self {
            llm_pool,
            config: config.unwrap_or_default(),
        }
    }

    pub async fn enrich(
        &self,
        request: KarmaEnrichmentRequest,
    ) -> Result<KarmaEnrichmentResponse, KarmaError> {
        if request.documents.is_empty() {
            return Err(KarmaError::new("No documents provided for enrichment"));
        }

        let mut logs = Vec::new();
        let mut graph = request.graph.clone();
        let mut new_nodes = Vec::new();
        let mut new_edges = Vec::new();
        let mut accepted = Vec::new();
        let mut rejected = Vec::new();

        let plan_call = self
            .planner_step(&graph, request.goal.clone(), request.instructions.clone())
            .await?;
        logs.push(plan_call.log);
        let plan = plan_call.parsed;

        for (doc_index, document) in request.documents.iter().enumerate() {
            let extractor_call = self
                .extractor_step(
                    &graph,
                    document,
                    request.goal.clone(),
                    request.instructions.clone(),
                )
                .await?;
            logs.push(extractor_call.log.clone());
            let mut extractor_output = extractor_call.parsed.unwrap_or_default();

            if extractor_output.candidates.len() > self.config.max_candidates_per_document {
                extractor_output
                    .candidates
                    .truncate(self.config.max_candidates_per_document);
            }

            for candidate in extractor_output.candidates {
                let validator_call = self
                    .validator_step(&graph, &candidate, document, doc_index)
                    .await?;
                logs.push(validator_call.log.clone());

                if let Some(decision) = validator_call.parsed {
                    match decision.verdict {
                        KarmaVerdict::Accept => {
                            let (subject_node, subject_created) = graph
                                .ensure_node(&decision.triple.subject, &decision.triple.metadata);
                            let (object_node, object_created) = graph
                                .ensure_node(&decision.triple.object, &decision.triple.metadata);

                            if subject_created {
                                new_nodes.push(subject_node.clone());
                            }

                            if object_created {
                                new_nodes.push(object_node.clone());
                            }

                            let edge = graph.add_edge(
                                &subject_node,
                                &object_node,
                                &decision.triple.predicate,
                                &decision.triple.metadata,
                            );
                            new_edges.push(edge);
                            accepted.push(decision.triple);
                        }
                        KarmaVerdict::Reject => {
                            rejected.push(KarmaRejectedCandidate {
                                candidate: decision.triple,
                                reason: decision.explanation,
                            });
                        }
                        KarmaVerdict::NeedsClarification => {
                            rejected.push(KarmaRejectedCandidate {
                                candidate: decision.triple,
                                reason: decision.explanation,
                            });
                        }
                    }
                } else {
                    rejected.push(KarmaRejectedCandidate {
                        candidate,
                        reason: "Validator agent returned an unparsable response".to_string(),
                    });
                }
            }
        }

        Ok(KarmaEnrichmentResponse {
            plan,
            updated_graph: graph,
            new_nodes,
            new_edges,
            accepted_candidates: accepted,
            rejected_candidates: rejected,
            agent_logs: logs,
        })
    }

    async fn planner_step(
        &self,
        graph: &KnowledgeGraph,
        goal: Option<String>,
        instructions: Option<String>,
    ) -> Result<KarmaPlannerCall, KarmaError> {
        let mut prompt = String::from(
            "You are KARMA's planning agent. You orchestrate knowledge graph enrichment.\n",
        );
        prompt.push_str("Review the current knowledge graph snapshot and derive a plan. Respond strictly in JSON with keys objective, tasks (array) and success_metrics (array).\n");
        if let Some(goal) = goal {
            prompt.push_str(&format!("Enrichment goal: {}\n", goal));
        }
        if let Some(instr) = instructions {
            prompt.push_str(&format!("Operator instructions: {}\n", instr));
        }
        prompt.push_str("Current graph snapshot (JSON):\n");
        prompt.push_str(&graph.as_prompt_context());

        let messages = self.build_messages(Some(&prompt), None, "Return the JSON plan now.");
        let call: KarmaAgentCall<KarmaPlannerOutput> =
            self.call_agent(KarmaAgentKind::Planner, messages).await?;

        Ok(KarmaPlannerCall {
            parsed: call.parsed,
            log: KarmaAgentLog {
                agent: KarmaAgentKind::Planner,
                prompt,
                response: call.raw,
                timestamp_ms: current_timestamp_ms(),
                parsed_successfully: call.parsed.is_some(),
            },
        })
    }

    async fn extractor_step(
        &self,
        graph: &KnowledgeGraph,
        document: &str,
        goal: Option<String>,
        instructions: Option<String>,
    ) -> Result<KarmaExtractorCall, KarmaError> {
        let mut prompt = String::from("You are KARMA's extraction agent.\n");
        prompt.push_str("Identify candidate triples that enrich the knowledge graph. Use the provided document and respond in JSON with fields candidates (array) and notes (string). Each candidate must include subject, predicate, object, justification, confidence (0-1), and metadata (object).\n");
        if let Some(goal) = goal {
            prompt.push_str(&format!("Goal: {}\n", goal));
        }
        if let Some(instr) = instructions {
            prompt.push_str(&format!("Additional instructions: {}\n", instr));
        }
        prompt.push_str("Current graph snapshot: ");
        prompt.push_str(&graph.as_prompt_context());
        prompt.push_str("\nDocument:\n");
        prompt.push_str(document);

        let messages = self.build_messages(
            Some(&prompt),
            None,
            "Return only JSON following the schema.",
        );
        let call: KarmaAgentCall<KarmaExtractorOutput> =
            self.call_agent(KarmaAgentKind::Extractor, messages).await?;

        Ok(KarmaExtractorCall {
            parsed: call.parsed,
            log: KarmaAgentLog {
                agent: KarmaAgentKind::Extractor,
                prompt,
                response: call.raw,
                timestamp_ms: current_timestamp_ms(),
                parsed_successfully: call.parsed.is_some(),
            },
        })
    }

    async fn validator_step(
        &self,
        graph: &KnowledgeGraph,
        candidate: &KarmaCandidateTriple,
        document: &str,
        doc_index: usize,
    ) -> Result<KarmaValidatorCall, KarmaError> {
        let mut prompt = String::from("You are KARMA's validation agent.\n");
        prompt.push_str("Decide if the proposed triple is correct and should be added to the knowledge graph. Respond strictly in JSON with keys verdict (accept|reject|needs_clarification), confidence (0-1), explanation, and triple.\n");
        prompt.push_str("Current graph snapshot: ");
        prompt.push_str(&graph.as_prompt_context());
        prompt.push_str("\nCandidate triple JSON:\n");
        prompt.push_str(
            &serde_json::to_string_pretty(candidate).unwrap_or_else(|_| "{}".to_string()),
        );
        prompt.push_str("\nSupporting document #");
        prompt.push_str(&(doc_index + 1).to_string());
        prompt.push_str(":\n");
        prompt.push_str(document);

        let messages = self.build_messages(Some(&prompt), None, "Return the validation JSON now.");
        let call: KarmaAgentCall<KarmaValidatorDecision> =
            self.call_agent(KarmaAgentKind::Validator, messages).await?;

        Ok(KarmaValidatorCall {
            parsed: call.parsed,
            log: KarmaAgentLog {
                agent: KarmaAgentKind::Validator,
                prompt,
                response: call.raw,
                timestamp_ms: current_timestamp_ms(),
                parsed_successfully: call.parsed.is_some(),
            },
        })
    }

    fn build_messages(
        &self,
        system_prompt: Option<&str>,
        developer_prompt: Option<&str>,
        user_prompt: &str,
    ) -> Vec<Message> {
        let mut messages = Vec::new();
        if let Some(system) = system_prompt {
            messages.push(Message {
                role: Some(Role::System),
                content: Some(Content::String(system.to_string())),
            });
        }
        if let Some(developer) = developer_prompt {
            messages.push(Message {
                role: Some(Role::Developer),
                content: Some(Content::String(developer.to_string())),
            });
        }
        messages.push(Message {
            role: Some(Role::User),
            content: Some(Content::String(user_prompt.to_string())),
        });
        messages
    }

    async fn call_agent<T: for<'de> Deserialize<'de> + Send + 'static + Clone>(
        &self,
        agent: KarmaAgentKind,
        messages: Vec<Message>,
    ) -> Result<KarmaAgentCall<T>, KarmaError> {
        let mut rng = rand::rng();
        let Some(llm) = self.llm_pool.choose(&mut rng) else {
            return Err(KarmaError::new("Model pool is empty"));
        };
        let send_future = llm.send(ProcessMessages { messages });
        match actix_web::rt::time::timeout(
            Duration::from_secs(self.config.request_timeout_secs),
            send_future,
        )
        .await
        {
            Ok(Ok(Ok(stream))) => {
                let chunks = stream.collect::<Vec<_>>().await;
                let raw = chunks.join("");
                let parsed = parse_agent_json::<T>(&raw);
                Ok(KarmaAgentCall { raw, parsed })
            }
            Ok(Ok(Err(_))) => Err(KarmaError::new(format!(
                "{} agent returned an empty stream",
                agent
            ))),
            Ok(Err(e)) => Err(KarmaError::new(format!(
                "{} agent mailbox error: {}",
                agent, e
            ))),
            Err(_) => Err(KarmaError::new(format!(
                "{} agent timed out after {} seconds",
                agent, self.config.request_timeout_secs
            ))),
        }
    }
}

#[derive(Debug, Clone)]
struct KarmaPlannerCall {
    parsed: Option<KarmaPlannerOutput>,
    log: KarmaAgentLog,
}

#[derive(Debug, Clone)]
struct KarmaExtractorCall {
    parsed: Option<KarmaExtractorOutput>,
    log: KarmaAgentLog,
}

#[derive(Debug, Clone)]
struct KarmaValidatorCall {
    parsed: Option<KarmaValidatorDecision>,
    log: KarmaAgentLog,
}

#[derive(Debug)]
pub struct KarmaError {
    message: String,
}

impl KarmaError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for KarmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for KarmaError {}

fn parse_agent_json<T: for<'de> Deserialize<'de>>(raw: &str) -> Option<T> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }

    let candidate = if trimmed.starts_with('{') || trimmed.starts_with('[') {
        Some(trimmed.to_string())
    } else if let Some(start) = trimmed.find('{') {
        let end = trimmed.rfind('}')?;
        Some(trimmed[start..=end].to_string())
    } else if let Some(start) = trimmed.find('[') {
        let end = trimmed.rfind(']')?;
        Some(trimmed[start..=end].to_string())
    } else {
        None
    }?;

    let sanitized = strip_json_code_fence(&candidate);
    serde_json::from_str::<T>(&sanitized).ok()
}

fn strip_json_code_fence(input: &str) -> String {
    let trimmed = input.trim();
    if trimmed.starts_with("```") {
        let mut lines = trimmed.lines();
        lines.next();
        let mut collected = Vec::new();
        for line in lines {
            if line.trim_start().starts_with("```") {
                break;
            }
            collected.push(line);
        }
        collected.join("\n")
    } else {
        trimmed.to_string()
    }
}

fn current_timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or_default()
}

#[utoipa::path(
    request_body = KarmaEnrichmentRequest,
    responses(
        (status = OK, description = "Success", body = KarmaEnrichmentResponse, content_type = "application/json")
    ),
    security(
        ("api_key" = [])
    ),
)]
#[post("/knowledge/karma/enrich")]
pub async fn karma_enrich(
    body: Json<KarmaEnrichmentRequest>,
    llm_pool: web::Data<HashMap<String, Vec<Recipient<ProcessMessages>>>>,
) -> impl Responder {
    let request = body.into_inner();

    if request.model.trim().is_empty() {
        return HttpResponse::BadRequest().json(OpenAiError {
            message: "Model name is required".to_string(),
            code: "model_not_found".to_string(),
            r#type: "invalid_request_error".to_string(),
            param: None,
        });
    }

    let Some(model_pool) = llm_pool.get(&request.model) else {
        return HttpResponse::BadRequest().json(OpenAiError {
            message: format!(
                "The model {} does not exist or you do not have access to it.",
                request.model
            ),
            code: "model_not_found".to_string(),
            r#type: "invalid_request_error".to_string(),
            param: None,
        });
    };

    let orchestrator = KarmaOrchestrator::new(model_pool.clone(), None);

    match orchestrator.enrich(request).await {
        Ok(response) => HttpResponse::Ok().json(response),
        Err(err) => HttpResponse::BadRequest().json(OpenAiError {
            message: err.to_string(),
            code: "karma_enrichment_failed".to_string(),
            r#type: "invalid_request_error".to_string(),
            param: None,
        }),
    }
}
