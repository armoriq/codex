//! Coverage tests for `pre_tool_use_payload` on handlers that previously
//! relied on the trait's `None` default and therefore did not emit PreToolUse
//! hooks. See https://github.com/openai/codex/issues/20204.

use std::sync::Arc;

use codex_protocol::models::SearchToolCallParams;
use serde_json::Value;
use serde_json::json;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::session::session::Session;
use crate::session::tests::make_session_and_context;
use crate::session::turn_context::TurnContext;
use crate::tools::context::ToolCallSource;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::hook_names::HookToolName;
use crate::tools::registry::CoreToolRuntime;
use crate::tools::registry::PreToolUsePayload;
use crate::turn_diff_tracker::TurnDiffTracker;

use super::agent_jobs::SpawnAgentsOnCsvHandler;
use super::goal::CreateGoalHandler;
use super::goal::GetGoalHandler;
use super::mcp_resource::ListMcpResourcesHandler;
use super::plan::PlanHandler;
use super::tool_search::ToolSearchHandler;
use super::view_image::ViewImageHandler;

fn invocation_with_payload(
    session: Session,
    turn: TurnContext,
    tool_name: &str,
    call_id: &str,
    payload: ToolPayload,
) -> ToolInvocation {
    ToolInvocation {
        session: Arc::new(session),
        turn: Arc::new(turn),
        cancellation_token: CancellationToken::new(),
        tracker: Arc::new(Mutex::new(TurnDiffTracker::new())),
        call_id: call_id.to_string(),
        tool_name: codex_tools::ToolName::plain(tool_name),
        source: ToolCallSource::Direct,
        payload,
    }
}

fn function_payload(args: &Value) -> ToolPayload {
    ToolPayload::Function {
        arguments: args.to_string(),
    }
}

#[tokio::test]
async fn plan_handler_emits_pre_tool_use_payload() {
    let (session, turn) = make_session_and_context().await;
    let args = json!({ "explanation": "test", "plan": [] });
    let invocation = invocation_with_payload(
        session,
        turn,
        "update_plan",
        "call-plan",
        function_payload(&args),
    );

    assert_eq!(
        PlanHandler.pre_tool_use_payload(&invocation),
        Some(PreToolUsePayload {
            tool_name: HookToolName::new("update_plan"),
            tool_input: args,
        })
    );
}

#[tokio::test]
async fn view_image_handler_emits_pre_tool_use_payload() {
    let (session, turn) = make_session_and_context().await;
    let args = json!({ "path": "/tmp/example.png" });
    let invocation = invocation_with_payload(
        session,
        turn,
        "view_image",
        "call-view-image",
        function_payload(&args),
    );

    assert_eq!(
        ViewImageHandler::default().pre_tool_use_payload(&invocation),
        Some(PreToolUsePayload {
            tool_name: HookToolName::new("view_image"),
            tool_input: args,
        })
    );
}

#[tokio::test]
async fn create_goal_handler_emits_pre_tool_use_payload() {
    let (session, turn) = make_session_and_context().await;
    let args = json!({ "goal": "ship pretooluse hooks" });
    let invocation = invocation_with_payload(
        session,
        turn,
        "create_goal",
        "call-create-goal",
        function_payload(&args),
    );

    assert_eq!(
        CreateGoalHandler.pre_tool_use_payload(&invocation),
        Some(PreToolUsePayload {
            tool_name: HookToolName::new("create_goal"),
            tool_input: args,
        })
    );
}

#[tokio::test]
async fn get_goal_handler_falls_back_to_raw_arguments_when_non_json() {
    let (session, turn) = make_session_and_context().await;
    let invocation = invocation_with_payload(
        session,
        turn,
        "get_goal",
        "call-get-goal",
        ToolPayload::Function {
            arguments: "not-json".to_string(),
        },
    );

    assert_eq!(
        GetGoalHandler.pre_tool_use_payload(&invocation),
        Some(PreToolUsePayload {
            tool_name: HookToolName::new("get_goal"),
            tool_input: json!({ "raw_arguments": "not-json" }),
        })
    );
}

#[tokio::test]
async fn list_mcp_resources_handler_emits_pre_tool_use_payload() {
    let (session, turn) = make_session_and_context().await;
    let args = json!({ "server": "test-mcp" });
    let invocation = invocation_with_payload(
        session,
        turn,
        "list_mcp_resources",
        "call-list-mcp-resources",
        function_payload(&args),
    );

    assert_eq!(
        ListMcpResourcesHandler.pre_tool_use_payload(&invocation),
        Some(PreToolUsePayload {
            tool_name: HookToolName::new("list_mcp_resources"),
            tool_input: args,
        })
    );
}

#[tokio::test]
async fn spawn_agents_on_csv_handler_emits_pre_tool_use_payload() {
    let (session, turn) = make_session_and_context().await;
    let args = json!({
        "csv_path": "/tmp/in.csv",
        "instruction": "process row {name}",
    });
    let invocation = invocation_with_payload(
        session,
        turn,
        "spawn_agents_on_csv",
        "call-spawn-agents",
        function_payload(&args),
    );

    assert_eq!(
        SpawnAgentsOnCsvHandler.pre_tool_use_payload(&invocation),
        Some(PreToolUsePayload {
            tool_name: HookToolName::new("spawn_agents_on_csv"),
            tool_input: args,
        })
    );
}

#[tokio::test]
async fn tool_search_handler_emits_query_and_limit() {
    let (session, turn) = make_session_and_context().await;
    let invocation = invocation_with_payload(
        session,
        turn,
        codex_tools::TOOL_SEARCH_TOOL_NAME,
        "call-tool-search",
        ToolPayload::ToolSearch {
            arguments: SearchToolCallParams {
                query: "list files".to_string(),
                limit: Some(5),
            },
        },
    );

    let handler = ToolSearchHandler::new(Vec::new());
    assert_eq!(
        handler.pre_tool_use_payload(&invocation),
        Some(PreToolUsePayload {
            tool_name: HookToolName::new(codex_tools::TOOL_SEARCH_TOOL_NAME),
            tool_input: json!({ "query": "list files", "limit": 5 }),
        })
    );
}

#[tokio::test]
async fn plan_handler_returns_none_for_non_function_payload() {
    let (session, turn) = make_session_and_context().await;
    let invocation = invocation_with_payload(
        session,
        turn,
        "update_plan",
        "call-plan",
        ToolPayload::Custom {
            input: "raw".to_string(),
        },
    );

    assert_eq!(PlanHandler.pre_tool_use_payload(&invocation), None);
}
