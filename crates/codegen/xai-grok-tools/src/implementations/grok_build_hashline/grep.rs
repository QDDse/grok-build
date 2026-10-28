//! `hashline_grep` — anchor-annotated search results.
//!
//! Delegates to the standard `GrepTool` for ripgrep execution, then
//! post-processes content-mode output to inject scheme-aware anchors.
//! Enables grep → edit workflows without an intermediate file read.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::computer::types::AsyncFileSystem;
use crate::implementations::grok_build::grep::{
    GrepParams, GrepSearchInput, GrepTool, OutputMode, RawSortedGrepResult,
};

use crate::types::output::{GrepFileMatch, GrepLineMatch, GrepSearchOutput};
use crate::types::requirements::{Expr, ToolRequirement};
use crate::types::resources::Params;
use crate::types::tool::{ToolKind, ToolNamespace};

use super::anchor::split_lines;
use super::config::HashlineSchemeParams;
use super::scheme::{Anchor, AnchorScheme};

/// Default timeout for anchor injection (seconds).
const DEFAULT_ANCHOR_TIMEOUT_SECS: u64 = 60;

/// Get cached anchors or generate and cache them for a single invocation.
async fn get_or_generate<'a>(
    cache: &'a mut HashMap<PathBuf, Vec<Anchor>>,
    path: &Path,
    scheme: &dyn AnchorScheme,
    fs: &dyn AsyncFileSystem,
) -> Option<&'a [Anchor]> {
    if !cache.contains_key(path) {
        let bytes = fs.read_file(path).await.ok()?;
        let content = String::from_utf8_lossy(&bytes);
        let lines = split_lines(&content);
        cache.insert(path.to_path_buf(), scheme.generate_anchors(&lines));
    }
    cache.get(path).map(|v| v.as_slice())
}

/// Inject anchors into ripgrep content-mode output. Transforms lines like `123: let x = 1;` or
/// `124- let y = 2;` into `123:abc:rst: let x = 1;` or `124:abc:rst- let y = 2;`. Lines that are
/// file headers or separators pass through unchanged.
#[cfg(test)]
pub(crate) async fn inject_anchors(
    stdout_bytes: &[u8],
    cwd: &Path,
    fs: &dyn AsyncFileSystem,
    scheme: &dyn AnchorScheme,
) -> Vec<u8> {
    let stdout = String::from_utf8_lossy(stdout_bytes);

    let (prefix, body, suffix) = match (stdout.find(">\n"), stdout.rfind("\n</workspace_result>")) {
        (Some(start), Some(end)) => {
            let body_start = start + 2;
            match (
                stdout.get(..body_start),
                stdout.get(body_start..end),
                stdout.get(end..),
            ) {
                (Some(prefix), Some(body), Some(suffix)) => (prefix, body, suffix),
                _ => return stdout_bytes.to_vec(),
            }
        }
        _ => return stdout_bytes.to_vec(),
    };

    let mut file_anchors: HashMap<PathBuf, Vec<Anchor>> = HashMap::new();
    let mut result = String::from(prefix);
    let mut current_file: Option<PathBuf> = None;

    for line in body.lines() {
        if !result.ends_with('\n') && !result.ends_with('>') {
            result.push('\n');
        }

        // Group separators (--) pass through.
        if line == "--" {
            result.push_str(line);
            continue;
        }

        // Try to parse as a numbered match/context line first. This correctly handles file paths
        // that start with digits (e.g. "2024_migration.rs") — they won't parse as valid rg lines
        // because they lack a ':' or '-' separator after the numeric prefix.
        if let Some((line_num, separator, content)) = parse_rg_line(line)
            && let Some(ref file_path) = current_file
            && let Some(anchors) = get_or_generate(&mut file_anchors, file_path, scheme, fs).await
            && line_num.saturating_sub(1) < anchors.len()
        {
            let Some(a) = line_num.checked_sub(1).and_then(|i| anchors.get(i)) else {
                result.push_str(line);
                continue;
            };
            let suffix_str = match &a.context {
                Some(ctx) => format!("{}:{ctx}", a.local),
                None => a.local.clone(),
            };
            result.push_str(&format!("{line_num}:{suffix_str}{separator}{content}"));
            continue;
        }

        // Not a numbered line — could be a file header, summary, or
        // a numbered line that failed anchor lookup (passed through below).
        if parse_rg_line(line).is_none()
            && !line.starts_with("Found ")
            && !line.starts_with("... [")
            && !line.is_empty()
        {
            current_file = Some(cwd.join(line));
        }

        result.push_str(line);
    }

    result.push_str(suffix);
    result.into_bytes()
}

/// Parse a ripgrep numbered line: `123:content` or `45-context`.
fn parse_rg_line(line: &str) -> Option<(usize, char, &str)> {
    let bytes = line.as_bytes();
    let mut idx = 0;
    while bytes.get(idx).is_some_and(|b| b.is_ascii_digit()) {
        idx += 1;
    }
    if idx == 0 || idx >= bytes.len() {
        return None;
    }
    let sep = *bytes.get(idx)? as char;
    if sep != ':' && sep != '-' {
        return None;
    }
    let num: usize = line.get(..idx)?.parse().ok()?;
    Some((num, sep, line.get(idx + 1..)?))
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub struct HashlineGrepInput {
    #[serde(flatten)]
    pub grep: GrepSearchInput,
    /// Maximum number of matching lines, independent of file headers and context.
    /// When present, this takes precedence over the legacy raw-line `head_limit`.
    #[schemars(
        description = "Maximum matching lines, excluding headers and context. Takes precedence over head_limit; capped at 2000."
    )]
    pub max_matches: Option<usize>,
    /// Hard limit for the complete final stdout, including anchors and wrapper.
    pub max_output_bytes: Option<usize>,
}

struct RawLine {
    number: usize,
    separator: char,
    content: String,
}

struct SearchFile {
    path: String,
    lines: Vec<RawLine>,
}

struct DisplayLine {
    number: usize,
    is_match: bool,
    content: String,
    rendered: String,
}

struct DisplayFile {
    path: String,
    lines: Vec<DisplayLine>,
}

fn parse_search_files(raw: &[u8]) -> Vec<SearchFile> {
    let mut files: Vec<SearchFile> = Vec::new();
    let text = String::from_utf8_lossy(raw);
    let mut current: Option<usize> = None;
    for line in text.lines() {
        if line.is_empty() {
            current = None;
        } else if line == "--" {
            continue;
        } else if let Some(index) = current {
            if let Some((number, separator, content)) = parse_rg_line(line) {
                if let Some(file) = files.get_mut(index) {
                    file.lines.push(RawLine {
                        number,
                        separator,
                        content: content.to_owned(),
                    });
                }
            } else {
                files.push(SearchFile {
                    path: line.to_owned(),
                    lines: Vec::new(),
                });
                current = Some(files.len() - 1);
            }
        } else {
            files.push(SearchFile {
                path: line.to_owned(),
                lines: Vec::new(),
            });
            current = Some(files.len() - 1);
        }
    }
    files.sort_by(|a, b| a.path.cmp(&b.path));
    for file in &mut files {
        file.lines
            .sort_by_key(|line| (line.number, line.separator != ':'));
        file.lines.dedup_by_key(|line| line.number);
    }
    files
}

fn render_card(
    files: &[DisplayFile],
    cwd_display: &str,
    match_cap: usize,
    context_cap: usize,
    observed_matches: usize,
    raw_truncated: bool,
    before: usize,
    after: usize,
) -> (String, Vec<GrepFileMatch>, usize) {
    let mut body = String::new();
    let mut structured = Vec::new();
    let mut visible = 0;
    let mut context_seen = 0;
    for file in files {
        let retained: Vec<usize> = file
            .lines
            .iter()
            .filter(|line| line.is_match)
            .map(|line| line.number)
            .take(match_cap.saturating_sub(visible))
            .collect();
        if retained.is_empty() {
            continue;
        }
        let mut file_body = String::new();
        let mut matches = Vec::new();
        let mut previous_number: Option<usize> = None;
        for line in &file.lines {
            let included_match = line.is_match && retained.contains(&line.number);
            let within_context = !line.is_match
                && retained.iter().any(|&number| {
                    line.number >= number.saturating_sub(before)
                        && line.number <= number.saturating_add(after)
                });
            let included_context = if within_context {
                let include = context_seen < context_cap;
                context_seen += 1;
                include
            } else {
                false
            };
            if !included_match && !included_context {
                continue;
            }
            if !file_body.is_empty() {
                file_body.push('\n');
            }
            if previous_number.is_some_and(|previous| line.number > previous.saturating_add(1)) {
                file_body.push_str("--\n");
            }
            file_body.push_str(&line.rendered);
            previous_number = Some(line.number);
            if included_match {
                visible += 1;
                matches.push(GrepLineMatch {
                    line_number: line.number,
                    content: line.content.clone(),
                });
            }
        }
        if !matches.is_empty() {
            if !body.is_empty() {
                body.push('\n');
            }
            body.push_str(&file.path);
            body.push('\n');
            body.push_str(&file_body);
            structured.push(GrepFileMatch {
                path: file.path.clone(),
                matches,
            });
        }
    }
    let omitted = observed_matches.saturating_sub(visible);
    let summary = if observed_matches == 0 && !raw_truncated {
        "No matches found".to_owned()
    } else if omitted == 0 && !raw_truncated {
        format!("Found {visible} matching lines")
    } else if raw_truncated {
        format!("Showing {visible} of at least {observed_matches} matching lines")
    } else {
        format!("Showing {visible} of {observed_matches} matching lines")
    };
    let footer = if raw_truncated {
        if omitted > 0 {
            format!("... [at least {omitted} matching lines omitted] ...")
        } else {
            "... [raw search truncated; more matches may exist] ...".to_owned()
        }
    } else if omitted > 0 {
        format!(
            "... [{omitted} matching line{} omitted] ...",
            if omitted == 1 { "" } else { "s" }
        )
    } else {
        String::new()
    };
    let mut card = format!("<workspace_result workspace_path=\"{cwd_display}\">\n{summary}");
    if !body.is_empty() {
        card.push('\n');
        card.push_str(&body);
    }
    if !footer.is_empty() {
        card.push('\n');
        card.push_str(&footer);
    }
    card.push_str("\n</workspace_result>");
    (card, structured, visible)
}

#[allow(clippy::too_many_arguments)]
async fn render_budgeted_content(
    raw: &[u8],
    raw_truncated: bool,
    cwd: &Path,
    cwd_display: &str,
    fs: &dyn AsyncFileSystem,
    scheme: &dyn AnchorScheme,
    max_matches: usize,
    budget: usize,
    max_chars_per_line: usize,
    before: usize,
    after: usize,
) -> GrepSearchOutput {
    // A raw safety cap may stop inside a numbered line. Never turn that
    // partial fragment into a visible, anchored match.
    let raw = if raw_truncated && !raw.ends_with(b"\n") {
        raw.iter()
            .rposition(|&byte| byte == b'\n')
            .and_then(|index| raw.get(..=index))
            .unwrap_or(&[])
    } else {
        raw
    };
    let mut files = Vec::new();
    let mut cache = HashMap::new();
    let mut selected = 0;
    let mut observed = 0;
    for file in parse_search_files(raw) {
        let matching = file
            .lines
            .iter()
            .filter(|line| line.separator == ':')
            .count();
        observed += matching;
        if selected >= max_matches || matching == 0 {
            continue;
        }
        let path = cwd.join(&file.path);
        let Some(anchors) = get_or_generate(&mut cache, &path, scheme, fs).await else {
            continue;
        };
        let mut lines = Vec::new();
        for line in file.lines {
            if line.separator == ':' && selected >= max_matches {
                continue;
            }
            let Some(anchor) = line
                .number
                .checked_sub(1)
                .and_then(|index| anchors.get(index))
            else {
                continue;
            };
            let anchor_text = match &anchor.context {
                Some(context) => format!("{}:{context}", anchor.local),
                None => anchor.local.clone(),
            };
            let content = crate::implementations::grok_build::grep::trim_line(
                &line.content,
                max_chars_per_line,
            );
            let is_match = line.separator == ':';
            if is_match {
                selected += 1;
            }
            lines.push(DisplayLine {
                number: line.number,
                is_match,
                rendered: format!("{}:{anchor_text}{}{content}", line.number, line.separator),
                content,
            });
        }
        if lines.iter().any(|line| line.is_match) {
            files.push(DisplayFile {
                path: file.path,
                lines,
            });
        }
    }

    let fit = |match_cap, context_cap| {
        render_card(
            &files,
            cwd_display,
            match_cap,
            context_cap,
            observed,
            raw_truncated,
            before,
            after,
        )
    };
    let all_contexts = files
        .iter()
        .map(|file| file.lines.iter().filter(|line| !line.is_match).count())
        .sum();
    let (mut card, mut structured, mut visible) = fit(selected, all_contexts);
    if card.len() > budget {
        let (no_context, _, _) = fit(selected, 0);
        if no_context.len() <= budget {
            let mut low = 0;
            let mut high = all_contexts;
            while low < high {
                let mid = low + (high - low).div_ceil(2);
                if fit(selected, mid).0.len() <= budget {
                    low = mid;
                } else {
                    high = mid - 1;
                }
            }
            (card, structured, visible) = fit(selected, low);
        } else {
            let mut low = 0;
            let mut high = selected;
            while low < high {
                let mid = low + (high - low).div_ceil(2);
                if fit(mid, 0).0.len() <= budget {
                    low = mid;
                } else {
                    high = mid - 1;
                }
            }
            (card, structured, visible) = fit(low, 0);
        }
    }
    let too_small = card.len() > budget;
    let stdout = if !too_small {
        card.into_bytes()
    } else if budget >= "[truncated]".len() {
        b"[truncated]".to_vec()
    } else if budget > 0 {
        b"!".to_vec()
    } else {
        Vec::new()
    };
    if stdout == b"[truncated]" || stdout == b"!" || stdout.is_empty() {
        structured.clear();
        visible = 0;
    }
    GrepSearchOutput {
        stdout,
        stderr: if too_small {
            b"Hashline stdout budget is too small for a complete result card.".to_vec()
        } else {
            Vec::new()
        },
        exit_code: 0,
        match_count: visible,
        file_matches: structured,
    }
}

fn enforce_stdout_budget(mut output: GrepSearchOutput, budget: usize) -> GrepSearchOutput {
    if output.stdout.len() > budget {
        output.stdout = if budget >= "[truncated]".len() {
            b"[truncated]".to_vec()
        } else if budget > 0 {
            b"!".to_vec()
        } else {
            Vec::new()
        };
        output.match_count = 0;
        output.file_matches.clear();
        if !output.stderr.is_empty() {
            output.stderr.push(b'\n');
        }
        output
            .stderr
            .extend_from_slice(b"Hashline stdout budget is too small for a complete result card.");
    }
    output
}

fn legacy_raw_line_limit(raw: &[u8], limit: usize) -> (&[u8], bool) {
    if limit == 0 {
        return (&[], !raw.is_empty());
    }
    let mut lines = 0;
    for (index, &byte) in raw.iter().enumerate() {
        if byte == b'\n' {
            lines += 1;
            if lines == limit {
                let end = index + 1;
                return (raw.get(..end).unwrap_or(raw), end < raw.len());
            }
        }
    }
    (raw, false)
}

const DESCRIPTION: &str = r#"Search file contents with anchor-annotated results${%- if tools.by_kind.edit %} for use with ${{ tools.by_kind.edit }}${%- endif %}.

Match lines include anchors${%- if tools.by_kind.edit %} you can pass directly to ${{ tools.by_kind.edit }}${%- if tools.by_kind.read %} without needing to ${{ tools.by_kind.read }} the file first${%- endif %}${%- endif %}. This grep format keeps grep-style separators after the anchor: `:` for
match lines and `-` for context lines.

Content output format:

  {grep_match}    ← match (:)
  {grep_context}    ← context (-)

Usage:
- ${{ params.search.pattern }} is a regex: `log.*Error`, `function\s+\w+`, `TODO`
- Default output is anchored content matches (no output-mode selector)
- Use -A, -B, -C for context lines around matches
- Use max_matches to cap matching lines independently of context and headers
- max_output_bytes caps the complete returned stdout, including paths and anchors
- Only use '${{ params.search.type }}' or '${{ params.search.glob }}' when certain of the file type
- Results are capped; omitted matches are reported in the output"#;

/// `hashline_grep` — searches with anchor-annotated results.
#[derive(Debug, Default)]
pub struct HashlineGrepTool;

impl crate::types::tool_metadata::ToolMetadata for HashlineGrepTool {
    fn kind(&self) -> ToolKind {
        ToolKind::Search
    }

    fn tool_namespace(&self) -> ToolNamespace {
        ToolNamespace::GrokBuildHashline
    }

    fn lock_path_param(&self) -> Option<&'static str> {
        Some("path")
    }

    fn description_template(&self) -> &str {
        DESCRIPTION
    }

    fn versioned_definition(
        &self,
        _contract_version: Option<&str>,
        client_name: &str,
        description_override: Option<&str>,
        renderer: &crate::types::template_renderer::TemplateRenderer,
        param_map: &std::collections::HashMap<String, String>,
        input_schema: &serde_json::Value,
        effective_params: &serde_json::Value,
    ) -> crate::types::definition::ToolDefinition {
        let params: HashlineSchemeParams =
            serde_json::from_value(effective_params.clone()).unwrap_or_default();
        params.build_tool_definition(
            DESCRIPTION,
            client_name,
            description_override,
            renderer,
            param_map,
            input_schema,
        )
    }

    fn requires_expr(&self) -> Expr<ToolRequirement> {
        Expr::True
    }
}

impl xai_tool_runtime::Tool for HashlineGrepTool {
    type Args = HashlineGrepInput;
    type Output = GrepSearchOutput;

    fn id(&self) -> xai_tool_protocol::ToolId {
        xai_tool_protocol::ToolId::new("hashline_grep").expect("valid tool id")
    }

    fn description(
        &self,
        _ctx: &::xai_tool_runtime::ListToolsContext,
    ) -> xai_tool_types::ToolDescription {
        xai_tool_types::ToolDescription::new(
            "hashline_grep",
            crate::types::tool_metadata::ToolMetadata::sanitized_description_template(self),
        )
    }

    fn capabilities(&self) -> xai_tool_protocol::ToolCapabilities {
        xai_tool_protocol::ToolCapabilities {
            is_read_only: true,
            tool_scope: Some(xai_tool_protocol::ToolScope::Read),
            ..Default::default()
        }
    }

    #[tracing::instrument(
        name = "tool.hashline_grep",
        skip_all,
        fields(timed_out = tracing::field::Empty)
    )]
    async fn run(
        &self,
        ctx: xai_tool_runtime::ToolCallContext,
        input: HashlineGrepInput,
    ) -> Result<GrepSearchOutput, xai_tool_runtime::ToolError> {
        use crate::types::tool_metadata::shared_resources;
        let resources = shared_resources(&ctx)?;
        let configured_budget = {
            let res = resources.lock().await;
            res.get::<Params<GrepParams>>()
                .and_then(|params| params.0.max_output_bytes)
                .unwrap_or(crate::DEFAULT_TOOL_OUTPUT_BYTES)
        };
        let requested_budget = input.max_output_bytes.unwrap_or(configured_budget);
        let output_mode = input
            .grep
            .output_mode
            .clone()
            .unwrap_or(OutputMode::Content);
        let grep = GrepTool;
        let cwd = crate::types::tool_metadata::resolve_cwd(&ctx, &resources).await?;
        let call_id = xai_tool_protocol::ToolCallId::new_v7();
        let mut rt_ctx = xai_tool_runtime::ToolCallContext::new(call_id);
        rt_ctx.extensions.insert(resources.clone());
        rt_ctx.extensions.insert(xai_tool_runtime::Cwd(cwd.clone()));
        let raw = match grep.run_raw_sorted(&rt_ctx, &input.grep).await? {
            RawSortedGrepResult::Raw(raw) => raw,
            RawSortedGrepResult::Early(out) => {
                return Ok(enforce_stdout_budget(out, requested_budget));
            }
        };
        let budget = input
            .max_output_bytes
            .unwrap_or(raw.default_max_output_bytes);
        if !matches!(output_mode, OutputMode::Content) {
            return Ok(enforce_stdout_budget(
                raw.into_standard(&input.grep),
                budget,
            ));
        }
        let (raw_stdout, raw_truncated, max_matches) = if let Some(max_matches) = input.max_matches
        {
            (raw.stdout.as_slice(), raw.truncated, max_matches.min(2_000))
        } else {
            let (limited, cut) =
                legacy_raw_line_limit(&raw.stdout, input.grep.head_limit.unwrap_or(200).min(2_000));
            (limited, raw.truncated || cut, 2_000)
        };
        let (fs, scheme) = {
            let res = resources.lock().await;
            let fs = res
                .require::<crate::types::resources::FileSystem>()?
                .0
                .clone();
            let params = res
                .get::<Params<HashlineSchemeParams>>()
                .cloned()
                .unwrap_or_default();
            let scheme = params
                .0
                .build_scheme()
                .map_err(xai_tool_runtime::ToolError::invalid_arguments)?;
            (fs, scheme)
        };
        let context = input.grep.context.filter(|&n| n > 0);
        let before = input
            .grep
            .before_context
            .filter(|&n| n > 0)
            .or(context)
            .unwrap_or(0);
        let after = input
            .grep
            .after_context
            .filter(|&n| n > 0)
            .or(context)
            .unwrap_or(0);
        match tokio::time::timeout(
            Duration::from_secs(DEFAULT_ANCHOR_TIMEOUT_SECS),
            render_budgeted_content(
                raw_stdout,
                raw_truncated,
                &cwd,
                &raw.cwd_display,
                &*fs,
                &*scheme,
                max_matches,
                budget,
                raw.max_chars_per_line,
                before,
                after,
            ),
        )
        .await
        {
            Ok(mut result) => {
                if !raw.stderr.is_empty() {
                    if !result.stderr.is_empty() {
                        result.stderr.push(b'\n');
                    }
                    result.stderr.extend_from_slice(&raw.stderr);
                }
                result.exit_code = raw.exit_code;
                Ok(result)
            }
            Err(_) => {
                tracing::Span::current().record("timed_out", true);
                let notice = "Hashline anchoring timed out; read the file before editing.";
                let stdout = if budget >= notice.len() {
                    notice.as_bytes().to_vec()
                } else if budget > 0 {
                    b"!".to_vec()
                } else {
                    Vec::new()
                };
                Ok(GrepSearchOutput {
                    stdout,
                    stderr: raw.stderr,
                    exit_code: -1,
                    match_count: 0,
                    file_matches: Vec::new(),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_scheme() -> Box<dyn AnchorScheme> {
        HashlineSchemeParams::default().build_scheme().unwrap()
    }

    #[test]
    fn parse_rg_line_match() {
        let (num, sep, content) = parse_rg_line("123:    let x = 1;").unwrap();
        assert_eq!(num, 123);
        assert_eq!(sep, ':');
        assert_eq!(content, "    let x = 1;");
    }

    #[test]
    fn parse_rg_line_context() {
        let (num, sep, content) = parse_rg_line("45-    let y = 2;").unwrap();
        assert_eq!(num, 45);
        assert_eq!(sep, '-');
        assert_eq!(content, "    let y = 2;");
    }

    #[test]
    fn parse_rg_line_no_digits() {
        assert!(parse_rg_line("src/main.rs").is_none());
    }

    #[test]
    fn parse_rg_line_empty() {
        assert!(parse_rg_line("").is_none());
    }

    #[test]
    fn tool_metadata() {
        use crate::types::tool_metadata::ToolMetadata;
        let tool = HashlineGrepTool;
        assert_eq!(xai_tool_runtime::Tool::id(&tool).as_str(), "hashline_grep");
        assert_eq!(ToolMetadata::kind(&tool), ToolKind::Search);
        assert!(xai_tool_runtime::Tool::capabilities(&tool).is_read_only);
        assert!(matches!(
            ToolMetadata::tool_namespace(&tool),
            ToolNamespace::GrokBuildHashline
        ));
    }

    #[test]
    fn description_template_references_edit_tool() {
        use crate::types::tool_metadata::ToolMetadata;
        let tool = HashlineGrepTool;
        assert!(ToolMetadata::description_template(&tool).contains("tools.by_kind.edit"));
    }

    #[test]
    fn hashline_input_keeps_legacy_grep_fields_and_exposes_new_limits() {
        let input: HashlineGrepInput = serde_json::from_value(serde_json::json!({
            "pattern": "needle", "-B": 2, "head_limit": 7,
            "max_matches": 3, "max_output_bytes": 256
        }))
        .unwrap();
        assert_eq!(input.grep.pattern, "needle");
        assert_eq!(input.grep.before_context, Some(2));
        assert_eq!(input.grep.head_limit, Some(7));
        assert_eq!(input.max_matches, Some(3));
        assert_eq!(input.max_output_bytes, Some(256));
        let schema = crate::registry::types::generate_schema::<HashlineGrepInput>();
        let props = schema
            .get("properties")
            .and_then(|p| p.as_object())
            .unwrap();
        assert!(props.contains_key("pattern"));
        assert!(props.contains_key("-B"));
        assert!(props.contains_key("max_matches"));
        assert!(props.contains_key("max_output_bytes"));
    }

    #[test]
    fn legacy_head_limit_still_counts_raw_heading_and_context_lines() {
        let raw = b"a.rs\n1-before\n2:needle\n3-after\n";
        let (limited, truncated) = legacy_raw_line_limit(raw, 2);
        assert_eq!(limited, b"a.rs\n1-before\n");
        assert!(truncated);
        let (full, truncated) = legacy_raw_line_limit(raw, 4);
        assert_eq!(full, raw);
        assert!(!truncated);
    }

    #[tokio::test]
    async fn inject_anchors_content_mode() {
        use crate::computer::local::LocalFs;
        use std::sync::Arc;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("test.rs"),
            "fn main() {\n    let x = 1;\n    let y = 2;\n}\n",
        )
        .unwrap();

        let fs = Arc::new(LocalFs);

        // Simulate ripgrep output for "let" search.
        let rg_output = format!(
            "<workspace_result workspace_path=\"{}\">\n\
             Found 2 matching lines\n\
             test.rs\n\
             2:    let x = 1;\n\
             3:    let y = 2;\n\
             </workspace_result>",
            tmp.path().display()
        );

        let scheme = test_scheme();
        let result = inject_anchors(rg_output.as_bytes(), tmp.path(), &*fs, &*scheme).await;
        let output = String::from_utf8_lossy(&result);

        // Anchored lines should have the pattern NUM:LOCAL:CONTEXT:CONTENT
        // (3 colons for chunk scheme: line:local:context:content).
        for line in output.lines() {
            if line.starts_with(|c: char| c.is_ascii_digit()) {
                let colon_count = line.matches(':').count();
                assert!(
                    colon_count >= 3,
                    "anchored line should have ≥3 colons, got {colon_count}: {line}"
                );
            }
        }
    }

    #[tokio::test]
    async fn inject_anchors_preserves_headers() {
        use crate::computer::local::LocalFs;
        use std::sync::Arc;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.rs"), "hello\nworld\n").unwrap();

        let fs = Arc::new(LocalFs);
        let rg_output = format!(
            "<workspace_result workspace_path=\"{}\">\n\
             Found 1 matching lines\n\
             a.rs\n\
             1:hello\n\
             </workspace_result>",
            tmp.path().display()
        );

        let scheme = test_scheme();
        let result = inject_anchors(rg_output.as_bytes(), tmp.path(), &*fs, &*scheme).await;
        let output = String::from_utf8_lossy(&result);

        assert!(output.contains("Found 1 matching lines"));
        assert!(output.contains("a.rs\n"));
        assert!(output.contains("</workspace_result>"));
    }

    #[tokio::test]
    async fn inject_anchors_context_lines() {
        use crate::computer::local::LocalFs;
        use std::sync::Arc;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("test.rs"), "a\nb\nc\nd\ne\n").unwrap();

        let fs = Arc::new(LocalFs);
        let rg_output = format!(
            "<workspace_result workspace_path=\"{}\">\n\
             Found 1 matching lines\n\
             test.rs\n\
             2-b\n\
             3:c\n\
             4-d\n\
             </workspace_result>",
            tmp.path().display()
        );

        let scheme = test_scheme();
        let result = inject_anchors(rg_output.as_bytes(), tmp.path(), &*fs, &*scheme).await;
        let output = String::from_utf8_lossy(&result);

        // Context line should use '-' separator after the anchor.
        let line_2 = output.lines().find(|l| l.starts_with('2')).unwrap();
        assert!(
            line_2.contains('-'),
            "context line should keep '-': {line_2}"
        );

        // Match line should use ':' separator after the anchor.
        let line_3 = output.lines().find(|l| l.starts_with('3')).unwrap();
        // Count colons: line:local:context:content = 3 colons with ':'
        let colon_count = line_3.matches(':').count();
        assert!(
            colon_count >= 3,
            "match line should have ≥3 colons: {line_3}"
        );
    }

    #[tokio::test]
    async fn no_injection_for_missing_file() {
        use crate::computer::local::LocalFs;
        use std::sync::Arc;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        let fs = Arc::new(LocalFs);

        // File doesn't exist — anchors can't be generated.
        let rg_output = format!(
            "<workspace_result workspace_path=\"{}\">\n\
             Found 1 matching lines\n\
             nonexistent.rs\n\
             5:some content\n\
             </workspace_result>",
            tmp.path().display()
        );

        let scheme = test_scheme();
        let result = inject_anchors(rg_output.as_bytes(), tmp.path(), &*fs, &*scheme).await;
        let output = String::from_utf8_lossy(&result);

        // Should fall through without anchors — original line preserved.
        assert!(output.contains("5:some content"));
    }

    #[tokio::test]
    async fn digit_leading_file_path_recognized_as_header() {
        use crate::computer::local::LocalFs;
        use std::sync::Arc;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        let subdir = tmp.path().join("2024_data");
        std::fs::create_dir_all(&subdir).unwrap();
        std::fs::write(subdir.join("config.rs"), "let x = 1;\nlet y = 2;\n").unwrap();

        let fs = Arc::new(LocalFs);
        let rg_output = format!(
            "<workspace_result workspace_path=\"{}\">\n\
             Found 1 matching lines\n\
             2024_data/config.rs\n\
             1:let x = 1;\n\
             </workspace_result>",
            tmp.path().display()
        );

        let scheme = test_scheme();
        let result = inject_anchors(rg_output.as_bytes(), tmp.path(), &*fs, &*scheme).await;
        let output = String::from_utf8_lossy(&result);

        // The file header should be recognized despite starting with digits.
        // Line 1 should be annotated with anchors from the correct file.
        let line_1 = output.lines().find(|l| l.starts_with('1')).unwrap();
        let colon_count = line_1.matches(':').count();
        assert!(
            colon_count >= 3,
            "anchors should be from the correct file, got: {line_1}"
        );
    }

    #[tokio::test]
    async fn no_injection_for_files_with_matches_output() {
        use crate::computer::local::LocalFs;
        use std::sync::Arc;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        let fs = Arc::new(LocalFs);

        // files_with_matches output has no numbered lines — just file paths.
        let rg_output = format!(
            "<workspace_result workspace_path=\"{}\">\n\
             Found 2 files\n\
             src/main.rs\n\
             src/lib.rs\n\
             </workspace_result>",
            tmp.path().display()
        );

        let scheme = test_scheme();
        let result = inject_anchors(rg_output.as_bytes(), tmp.path(), &*fs, &*scheme).await;
        let output = String::from_utf8_lossy(&result);

        // No numbered lines → no anchor injection. Output should be unchanged.
        assert!(output.contains("src/main.rs"));
        assert!(output.contains("src/lib.rs"));
        assert!(
            !output.contains('→'),
            "files_with_matches should have no anchors"
        );
    }

    #[tokio::test]
    async fn no_injection_for_count_output() {
        use crate::computer::local::LocalFs;
        use std::sync::Arc;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        let fs = Arc::new(LocalFs);

        // count output: "file:N" format.
        let rg_output = format!(
            "<workspace_result workspace_path=\"{}\">\n\
             src/main.rs:5\n\
             src/lib.rs:3\n\
             </workspace_result>",
            tmp.path().display()
        );

        let scheme = test_scheme();
        let result = inject_anchors(rg_output.as_bytes(), tmp.path(), &*fs, &*scheme).await;
        let output = String::from_utf8_lossy(&result);

        // Count lines should pass through. They look like "src/main.rs:5"
        // which parse_rg_line won't match (path contains '/', not just digits).
        assert!(output.contains("src/main.rs:5"));
        assert!(output.contains("src/lib.rs:3"));
    }

    #[tokio::test]
    async fn multiple_matches_same_file_use_cached_anchors() {
        use crate::computer::local::LocalFs;
        use std::sync::Arc;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("multi.rs"),
            "fn a() {}\nfn b() {}\nfn c() {}\nfn d() {}\n",
        )
        .unwrap();

        let fs = Arc::new(LocalFs);
        let rg_output = format!(
            "<workspace_result workspace_path=\"{}\">\n\
             Found 4 matching lines\n\
             multi.rs\n\
             1:fn a() {{}}\n\
             2:fn b() {{}}\n\
             3:fn c() {{}}\n\
             4:fn d() {{}}\n\
             </workspace_result>",
            tmp.path().display()
        );

        let scheme = test_scheme();
        let result = inject_anchors(rg_output.as_bytes(), tmp.path(), &*fs, &*scheme).await;
        let output = String::from_utf8_lossy(&result);

        // All 4 lines should be anchored (same file, same cache entry).
        let anchored_count = output
            .lines()
            .filter(|l| l.starts_with(|c: char| c.is_ascii_digit()))
            .filter(|l| l.matches(':').count() >= 3)
            .count();
        assert_eq!(anchored_count, 4, "all 4 lines should be anchored");
    }

    #[tokio::test]
    async fn inject_anchors_timeout_returns_unanchored() {
        use crate::computer::local::LocalFs;
        use crate::computer::types::{AsyncFileSystem, ComputerError};
        use std::sync::Arc;
        use tempfile::TempDir;

        struct SlowFs;

        #[async_trait::async_trait]
        impl AsyncFileSystem for SlowFs {
            async fn read_file(&self, _path: &Path) -> Result<Vec<u8>, ComputerError> {
                tokio::time::sleep(Duration::from_secs(10)).await;
                Ok(Vec::new())
            }
            async fn write_file(&self, _path: &Path, _data: &[u8]) -> Result<(), ComputerError> {
                unimplemented!()
            }
            async fn delete_file(&self, _path: &Path) -> Result<(), ComputerError> {
                unimplemented!()
            }
        }

        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("test.rs"),
            "fn main() {\n    let x = 1;\n    let y = 2;\n}\n",
        )
        .unwrap();

        let rg_output = format!(
            "<workspace_result workspace_path=\"{}\">\n\
             Found 2 matching lines\n\
             test.rs\n\
             2:    let x = 1;\n\
             3:    let y = 2;\n\
             </workspace_result>",
            tmp.path().display()
        );
        let rg_bytes = rg_output.as_bytes();

        let scheme = test_scheme();

        // Happy path: real filesystem produces anchored output.
        let fast_fs = Arc::new(LocalFs);
        let anchored = inject_anchors(rg_bytes, tmp.path(), &*fast_fs, &*scheme).await;
        let anchored_str = String::from_utf8_lossy(&anchored);
        for line in anchored_str.lines() {
            if line.starts_with(|c: char| c.is_ascii_digit()) {
                assert!(
                    line.matches(':').count() >= 3,
                    "happy path should produce anchored lines, got: {line}"
                );
            }
        }

        // Timeout path: slow filesystem causes cancellation.
        let slow_fs = Arc::new(SlowFs);
        let timeout_result = tokio::time::timeout(
            Duration::from_millis(10),
            inject_anchors(rg_bytes, tmp.path(), &*slow_fs, &*scheme),
        )
        .await;

        assert!(
            timeout_result.is_err(),
            "inject_anchors should have timed out"
        );

        // On timeout, production keeps the original output unchanged.
        let unanchored_str = String::from_utf8_lossy(rg_bytes);
        for line in unanchored_str.lines() {
            if line.starts_with(|c: char| c.is_ascii_digit()) {
                assert_eq!(
                    line.matches(':').count(),
                    1,
                    "un-anchored line should have exactly 1 colon, got: {line}"
                );
            }
        }

        assert!(unanchored_str.contains("Found 2 matching lines"));
        assert!(unanchored_str.contains("test.rs"));
        assert!(unanchored_str.contains("</workspace_result>"));
        assert_ne!(anchored, rg_bytes, "anchored output should differ from raw");
    }

    #[tokio::test]
    async fn truncated_output_stays_well_formed() {
        use crate::computer::local::LocalFs;
        use std::sync::Arc;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("big.rs"), "match\n".repeat(100)).unwrap();

        let fs = Arc::new(LocalFs);
        // Simulate truncated output with "... [N lines truncated]" marker.
        let mut rg_lines = String::new();
        for i in 1..=10 {
            rg_lines.push_str(&format!("{i}:match\n"));
        }
        let rg_output = format!(
            "<workspace_result workspace_path=\"{}\">\n\
             Found at least 100 matching lines\n\
             big.rs\n\
             {rg_lines}\
             ... [at least 90 lines truncated] ...\n\
             </workspace_result>",
            tmp.path().display()
        );

        let scheme = test_scheme();
        let result = inject_anchors(rg_output.as_bytes(), tmp.path(), &*fs, &*scheme).await;
        let output = String::from_utf8_lossy(&result);

        // Truncation marker should be preserved.
        assert!(output.contains("... [at least 90 lines truncated] ..."));
        // The wrapper should be intact.
        assert!(output.contains("</workspace_result>"));
        // Visible lines should be anchored.
        let anchored = output
            .lines()
            .filter(|l| l.starts_with(|c: char| c.is_ascii_digit()))
            .filter(|l| l.matches(':').count() >= 3)
            .count();
        assert_eq!(anchored, 10);
    }

    async fn budgeted_fixture(
        raw: &str,
        root: &Path,
        max_matches: usize,
        budget: usize,
        before: usize,
        after: usize,
    ) -> GrepSearchOutput {
        use crate::computer::local::LocalFs;
        render_budgeted_content(
            raw.as_bytes(),
            false,
            root,
            &root.display().to_string(),
            &LocalFs,
            &*test_scheme(),
            max_matches,
            budget,
            1_000,
            before,
            after,
        )
        .await
    }

    #[tokio::test]
    async fn budgeted_selection_sorts_files_before_limiting_matches() {
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.rs"), "needle a\n").unwrap();
        std::fs::write(tmp.path().join("z.rs"), "needle z\n").unwrap();
        let forward = "a.rs\n1:needle a\n\nz.rs\n1:needle z\n";
        let reverse = "z.rs\n1:needle z\n\na.rs\n1:needle a\n";
        let a = budgeted_fixture(forward, tmp.path(), 1, 4_096, 0, 0).await;
        let b = budgeted_fixture(reverse, tmp.path(), 1, 4_096, 0, 0).await;
        assert_eq!(a.stdout, b.stdout);
        assert_eq!(a.match_count, 1);
        assert_eq!(a.file_matches.len(), 1);
        assert_eq!(a.file_matches[0].path, "a.rs");
        assert!(!String::from_utf8_lossy(&a.stdout).contains("needle z"));
    }

    #[tokio::test]
    async fn budgeted_matches_are_separate_from_context_and_neighbours_are_unique() {
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("a.rs"),
            "before\nneedle one\nneedle two\nafter\n",
        )
        .unwrap();
        let raw = "a.rs\n1-before\n2:needle one\n3:needle two\n4-after\n";
        let out = budgeted_fixture(raw, tmp.path(), 2, 4_096, 1, 1).await;
        let stdout = String::from_utf8(out.stdout).unwrap();
        assert_eq!(out.match_count, 2);
        assert_eq!(out.file_matches[0].matches.len(), 2);
        assert_eq!(stdout.matches("1:").count(), 1);
        assert_eq!(stdout.matches("4:").count(), 1);
        assert_eq!(stdout.matches("needle one").count(), 1);
        assert_eq!(stdout.matches("needle two").count(), 1);
    }

    #[tokio::test]
    async fn budget_counts_wrapper_paths_utf8_and_full_anchors() {
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        let name = format!("{}中文.rs", "deep/".repeat(12));
        let file = tmp.path().join(&name);
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, "中文 needle\n中文 needle\n").unwrap();
        let raw = format!("{name}\n1:中文 needle\n2:中文 needle\n");
        let full = budgeted_fixture(&raw, tmp.path(), 2, usize::MAX, 0, 0).await;
        let exact = budgeted_fixture(&raw, tmp.path(), 2, full.stdout.len(), 0, 0).await;
        assert_eq!(full.stdout, exact.stdout);
        let shorter = budgeted_fixture(&raw, tmp.path(), 2, full.stdout.len() - 1, 0, 0).await;
        assert!(shorter.stdout.len() < full.stdout.len());
        let text = String::from_utf8(shorter.stdout).unwrap();
        for line in text
            .lines()
            .filter(|line| line.starts_with("1:") || line.starts_with("2:"))
        {
            assert!(line.matches(':').count() >= 3, "partial anchor: {line}");
        }
        assert_eq!(
            shorter.match_count,
            shorter
                .file_matches
                .iter()
                .map(|f| f.matches.len())
                .sum::<usize>()
        );
    }

    #[tokio::test]
    async fn four_character_anchors_are_complete_at_exact_byte_limit() {
        use crate::computer::local::LocalFs;
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.rs"), "needle\n").unwrap();
        let scheme = HashlineSchemeParams {
            hash_len: 4,
            ..Default::default()
        }
        .build_scheme()
        .unwrap();
        let full = render_budgeted_content(
            b"a.rs\n1:needle\n",
            false,
            tmp.path(),
            &tmp.path().display().to_string(),
            &LocalFs,
            &*scheme,
            1,
            usize::MAX,
            1_000,
            0,
            0,
        )
        .await;
        let exact = render_budgeted_content(
            b"a.rs\n1:needle\n",
            false,
            tmp.path(),
            &tmp.path().display().to_string(),
            &LocalFs,
            &*scheme,
            1,
            full.stdout.len(),
            1_000,
            0,
            0,
        )
        .await;
        assert_eq!(full.stdout, exact.stdout);
        let text = String::from_utf8(exact.stdout).unwrap();
        let match_line = text.lines().find(|line| line.starts_with("1:")).unwrap();
        let parts: Vec<_> = match_line.split(':').collect();
        assert_eq!(parts[1].len(), 4);
        assert_eq!(parts[2].len(), 4);
    }

    #[tokio::test]
    async fn tiny_budget_never_emits_partial_utf8_or_inconsistent_match_metadata() {
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.rs"), "中文 needle\n").unwrap();
        for budget in 0..100 {
            let out = budgeted_fixture("a.rs\n1:中文 needle\n", tmp.path(), 1, budget, 0, 0).await;
            assert!(out.stdout.len() <= budget, "budget {budget}");
            assert!(String::from_utf8(out.stdout).is_ok(), "budget {budget}");
            assert_eq!(
                out.match_count,
                out.file_matches
                    .iter()
                    .map(|f| f.matches.len())
                    .sum::<usize>()
            );
        }
    }

    #[tokio::test]
    async fn budgeted_truncation_counts_hidden_matches() {
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.rs"), "needle\nneedle\nneedle\n").unwrap();
        let out = budgeted_fixture(
            "a.rs\n1:needle\n2:needle\n3:needle\n",
            tmp.path(),
            2,
            4_096,
            0,
            0,
        )
        .await;
        let text = String::from_utf8(out.stdout).unwrap();
        assert_eq!(out.match_count, 2);
        assert!(text.contains("1 matching line omitted"), "{text}");
        assert!(!text.contains("3:needle"));
    }

    #[tokio::test]
    async fn raw_capture_truncation_reports_lower_bound_without_inventing_matches() {
        use crate::computer::local::LocalFs;
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.rs"), "needle\n").unwrap();
        let output = render_budgeted_content(
            b"a.rs\n1:needle\n2:need",
            true,
            tmp.path(),
            &tmp.path().display().to_string(),
            &LocalFs,
            &*test_scheme(),
            10,
            4_096,
            1_000,
            0,
            0,
        )
        .await;
        let text = String::from_utf8(output.stdout).unwrap();
        assert_eq!(output.match_count, 1);
        assert!(
            text.contains("Showing 1 of at least 1 matching lines"),
            "{text}"
        );
        assert!(text.contains("more matches may exist"), "{text}");
        assert!(!text.contains("2:need"), "{text}");
    }

    #[tokio::test]
    async fn budgeted_tool_keeps_ignore_and_read_deny_boundaries() {
        use crate::computer::local::LocalFs;
        use crate::types::resources::{Cwd, DenyReadGlobs, FileSystem, Resources};
        use crate::types::tool_metadata::test_ctx;
        use std::sync::Arc;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("safe.txt"), "needle safe\n").unwrap();
        std::fs::write(tmp.path().join("ignored.txt"), "needle ignored\n").unwrap();
        std::fs::write(tmp.path().join("secret.pem"), "needle secret\n").unwrap();
        std::fs::write(tmp.path().join(".ignore"), "ignored.txt\n").unwrap();
        let mut resources = Resources::new();
        resources.insert(Cwd(tmp.path().to_path_buf()));
        resources.insert(FileSystem(Arc::new(LocalFs)));
        resources.insert(DenyReadGlobs(vec!["**/*.pem".to_owned()]));
        let input: HashlineGrepInput = serde_json::from_value(serde_json::json!({
            "pattern": "needle", "max_matches": 10, "max_output_bytes": 4096
        }))
        .unwrap();
        let output = xai_tool_runtime::Tool::run(
            &HashlineGrepTool,
            test_ctx(resources.into_shared()),
            input,
        )
        .await
        .unwrap();
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert_eq!(output.match_count, 1, "{stdout}");
        assert!(stdout.contains("needle safe"), "{stdout}");
        assert!(!stdout.contains("needle ignored"), "{stdout}");
        assert!(!stdout.contains("needle secret"), "{stdout}");
        assert_eq!(output.file_matches.len(), 1);
    }

    #[tokio::test]
    async fn max_matches_overrides_legacy_head_limit_without_losing_hits_to_context() {
        use crate::computer::local::LocalFs;
        use crate::types::resources::{Cwd, FileSystem, Resources};
        use crate::types::tool_metadata::test_ctx;
        use std::sync::Arc;
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("a.rs"),
            "before\nneedle one\nneedle two\nafter\n",
        )
        .unwrap();
        let mut resources = Resources::new();
        resources.insert(Cwd(tmp.path().to_path_buf()));
        resources.insert(FileSystem(Arc::new(LocalFs)));
        let input: HashlineGrepInput = serde_json::from_value(serde_json::json!({
            "pattern": "needle", "head_limit": 1, "max_matches": 2,
            "-C": 1, "max_output_bytes": 4096
        }))
        .unwrap();
        let output = xai_tool_runtime::Tool::run(
            &HashlineGrepTool,
            test_ctx(resources.into_shared()),
            input,
        )
        .await
        .unwrap();
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert_eq!(output.match_count, 2, "{stdout}");
        assert!(
            stdout.contains("needle one") && stdout.contains("needle two"),
            "{stdout}"
        );
        assert_eq!(stdout.matches("before").count(), 1);
        assert_eq!(stdout.matches("after").count(), 1);
    }

    #[tokio::test]
    async fn missing_path_error_respects_configured_stdout_budget() {
        use crate::types::resources::{Cwd, Resources};
        use crate::types::tool_metadata::test_ctx;
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        let mut resources = Resources::new();
        resources.insert(Cwd(tmp.path().to_path_buf()));
        resources.insert(Params(GrepParams {
            max_output_bytes: Some(1),
            max_chars_per_line: None,
        }));
        let input: HashlineGrepInput = serde_json::from_value(serde_json::json!({
            "pattern": "needle", "path": "missing"
        }))
        .unwrap();
        let output = xai_tool_runtime::Tool::run(
            &HashlineGrepTool,
            test_ctx(resources.into_shared()),
            input,
        )
        .await
        .unwrap();
        assert_eq!(output.exit_code, 2);
        assert!(output.stdout.len() <= 1);
        assert!(String::from_utf8_lossy(&output.stderr).contains("budget is too small"));
        assert_eq!(output.match_count, 0);
    }
}
