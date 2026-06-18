//! Lightweight HTML and Markdown stripping for the build-time ingestion path.
//!
//! Neither parser is full-spec - we strip the markup that affects tokenization
//! and keep the underlying text. Anything more ambitious belongs in a
//! dedicated ingestion pipeline.

/// Strip HTML tags, decode the common named/numeric entities, and collapse
/// surrounding whitespace. Comments and `<script>` / `<style>` bodies are
/// dropped entirely.
pub fn strip_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'<' {
            // Lookahead for <!-- comment -->
            if bytes[i..].starts_with(b"<!--") {
                if let Some(end) = find_subseq(&bytes[i + 4..], b"-->") {
                    i += 4 + end + 3;
                    continue;
                } else {
                    break;
                }
            }
            // Lookahead for <script>...</script> and <style>...</style>
            if let Some((tag, rest_start)) = peek_tag(&bytes[i..]) {
                if tag.eq_ignore_ascii_case(b"script") || tag.eq_ignore_ascii_case(b"style") {
                    let close = if tag.eq_ignore_ascii_case(b"script") {
                        b"</script" as &[u8]
                    } else {
                        b"</style"
                    };
                    if let Some(end) = find_subseq_ci(&bytes[i + rest_start..], close) {
                        // skip past the </tag ... >
                        let after = i + rest_start + end;
                        if let Some(close_gt) = bytes[after..].iter().position(|&b| b == b'>') {
                            out.push(' ');
                            i = after + close_gt + 1;
                            continue;
                        }
                    }
                    break;
                }
            }
            // Otherwise: drop until the matching '>'.
            if let Some(end) = bytes[i + 1..].iter().position(|&b| b == b'>') {
                out.push(' ');
                i += 1 + end + 1;
                continue;
            } else {
                break;
            }
        }
        if c == b'&' {
            if let Some((decoded, len)) = decode_entity(&bytes[i..]) {
                out.push_str(&decoded);
                i += len;
                continue;
            }
        }
        // Non-ASCII: walk forward by the UTF-8 char length so we emit the
        // codepoint intact instead of pushing each leading-bit byte as its
        // own (Latin-1-shaped) char.
        if c >= 0x80 {
            let width = utf8_char_width(c).max(1);
            let end = (i + width).min(bytes.len());
            if let Ok(s) = std::str::from_utf8(&bytes[i..end]) {
                out.push_str(s);
            }
            i = end;
            continue;
        }
        out.push(c as char);
        i += 1;
    }
    collapse_ws(&out)
}

/// UTF-8 byte width for the leading byte of a codepoint, per RFC 3629.
/// Returns 1 for any invalid lead so we make forward progress.
fn utf8_char_width(b: u8) -> usize {
    // < 0xC0 covers both ASCII (1-byte) and stray continuation bytes, which we
    // treat as a 1-byte advance to recover from malformed input.
    if b < 0xC0 {
        1
    } else if b < 0xE0 {
        2
    } else if b < 0xF0 {
        3
    } else if b < 0xF8 {
        4
    } else {
        1
    }
}

/// Strip a useful subset of CommonMark markup: fenced code blocks, inline
/// code spans, link/image syntax, emphasis markers, heading markers,
/// blockquotes, and list bullets.
pub fn strip_markdown(s: &str) -> String {
    let mut out = String::with_capacity(s.len());

    // 1. Drop fenced code blocks delimited by ``` or ~~~.
    let mut in_fence = false;
    let mut fence_char = b'`';
    for line in s.lines() {
        let trimmed = line.trim_start();
        let bytes = trimmed.as_bytes();
        let is_fence = bytes.len() >= 3
            && (bytes.starts_with(b"```") || bytes.starts_with(b"~~~"))
            && bytes[0..3].iter().all(|&b| b == bytes[0]);
        if is_fence {
            if !in_fence {
                in_fence = true;
                fence_char = bytes[0];
            } else if bytes[0] == fence_char {
                in_fence = false;
            }
            continue;
        }
        if in_fence {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }

    // 2. Per-line: drop list bullets, blockquote, heading markers.
    let stripped_lines: Vec<String> = out
        .lines()
        .map(|line| {
            let mut s = line.trim_start();
            // headings: leading #'s
            while s.starts_with('#') {
                s = &s[1..];
            }
            // blockquotes
            while let Some(rest) = s.strip_prefix('>') {
                s = rest.trim_start();
            }
            // list bullets: *, -, + or ordered "1." / "1)"
            if let Some(rest) = s.strip_prefix(|c| matches!(c, '*' | '-' | '+')) {
                if rest.starts_with(char::is_whitespace) {
                    s = rest.trim_start();
                }
            }
            // ordered list
            let mut chars = s.chars();
            let mut digits = 0;
            for ch in chars.by_ref() {
                if ch.is_ascii_digit() {
                    digits += 1;
                } else if digits > 0 && (ch == '.' || ch == ')') {
                    s = chars.as_str().trim_start();
                    break;
                } else {
                    break;
                }
            }
            s.trim_start().to_string()
        })
        .collect();
    let mut joined = stripped_lines.join("\n");

    // 3. Inline transforms.
    // Replace ![alt](url) with alt
    joined = replace_link(&joined, true);
    // Replace [text](url) with text
    joined = replace_link(&joined, false);
    // Inline code: `...` -> drop backticks
    joined = strip_chars(&joined, &['`']);
    // Emphasis: drop standalone * and _ around words. We intentionally leave
    // characters inside words alone (e.g. an underscore in a snake_case
    // identifier is meaningful) - only strip when they're flanked by ASCII
    // word chars and another emphasis marker.
    joined = strip_emphasis(&joined);

    collapse_ws(&joined)
}

/// Best-effort detection: if the text contains '<' followed by an ASCII tag
/// name, treat as HTML; if it contains markdown sigils on line starts, treat
/// as markdown; otherwise return as-is.
pub fn detect_and_clean(s: &str) -> String {
    let head: String = s.chars().take(2048).collect();
    let looks_html = head.contains("<p>")
        || head.contains("<div")
        || head.contains("<br")
        || head.contains("<html")
        || head.contains("<body")
        || head.contains("<script")
        || head.contains("</");
    if looks_html {
        return strip_html(s);
    }
    let looks_md = head.lines().any(|l| {
        let t = l.trim_start();
        t.starts_with("# ") || t.starts_with("```") || t.starts_with("* ") || t.starts_with("- ")
    });
    if looks_md {
        return strip_markdown(s);
    }
    s.to_string()
}

// ─── helpers ──────────────────────────────────────────────────────────

fn collapse_ws(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_ws = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !prev_ws {
                out.push(' ');
                prev_ws = true;
            }
        } else {
            out.push(c);
            prev_ws = false;
        }
    }
    out.trim().to_string()
}

fn find_subseq(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn find_subseq_ci(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len())
        .position(|w| w.eq_ignore_ascii_case(needle))
}

/// Given a slice starting at `<`, return the tag name and the offset just
/// after the tag-open token (i.e. past `<tag` but before attributes).
fn peek_tag(bytes: &[u8]) -> Option<(&[u8], usize)> {
    if bytes.first() != Some(&b'<') {
        return None;
    }
    let mut i = 1;
    while i < bytes.len() && (bytes[i] as char).is_ascii_alphanumeric() {
        i += 1;
    }
    if i == 1 {
        return None;
    }
    Some((&bytes[1..i], i))
}

fn decode_entity(bytes: &[u8]) -> Option<(String, usize)> {
    // bytes[0] == b'&'
    // Find ';' within a small window.
    let end = bytes[1..].iter().take(10).position(|&b| b == b';')?;
    let body = std::str::from_utf8(&bytes[1..1 + end]).ok()?;
    let total = 1 + end + 1;
    let s = match body {
        "amp" => "&".to_string(),
        "lt" => "<".to_string(),
        "gt" => ">".to_string(),
        "quot" => "\"".to_string(),
        "apos" => "'".to_string(),
        "nbsp" => " ".to_string(),
        _ if body.starts_with('#') => {
            let n = &body[1..];
            let code = if let Some(hex) = n.strip_prefix(|c: char| c == 'x' || c == 'X') {
                u32::from_str_radix(hex, 16).ok()?
            } else {
                n.parse::<u32>().ok()?
            };
            char::from_u32(code)?.to_string()
        }
        _ => return None,
    };
    Some((s, total))
}

fn replace_link(s: &str, image: bool) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.char_indices().peekable();
    let prefix = if image { '!' } else { '[' };
    while let Some((i, c)) = chars.next() {
        if image && c == '!' {
            if matches!(chars.peek(), Some(&(_, '['))) {
                if let Some((alt, rest_off)) = scan_link(&s[i + 1..]) {
                    out.push_str(&alt);
                    for _ in 0..rest_off {
                        chars.next();
                    }
                    continue;
                }
            }
            out.push(c);
        } else if !image && c == '[' && prefix == '[' {
            if let Some((text, rest_off)) = scan_link(&s[i..]) {
                out.push_str(&text);
                for _ in 0..rest_off - 1 {
                    chars.next();
                }
                continue;
            }
            out.push(c);
        } else {
            out.push(c);
        }
    }
    out
}

/// Given a slice starting at `[`, parse `[text](url)` and return `(text, consumed_byte_len)`.
fn scan_link(s: &str) -> Option<(String, usize)> {
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'[') {
        return None;
    }
    let close = bytes[1..].iter().position(|&b| b == b']')?;
    let text = &s[1..1 + close];
    let after = 1 + close + 1;
    if bytes.get(after) != Some(&b'(') {
        return None;
    }
    let close_paren = bytes[after + 1..].iter().position(|&b| b == b')')?;
    Some((text.to_string(), after + 1 + close_paren + 1))
}

fn strip_chars(s: &str, drop: &[char]) -> String {
    s.chars().filter(|c| !drop.contains(c)).collect()
}

fn strip_emphasis(s: &str) -> String {
    // Drop solo '*' and '_' characters when between word chars. Cheap heuristic:
    // we just drop all '*' and replace standalone '_' that don't sit between two
    // ascii word chars.
    let mut out = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if c == '*' {
            continue;
        }
        if c == '_' {
            let prev_is_word = i > 0 && (chars[i - 1].is_ascii_alphanumeric());
            let next_is_word = i + 1 < chars.len() && chars[i + 1].is_ascii_alphanumeric();
            if !(prev_is_word && next_is_word) {
                continue;
            }
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_basic() {
        let s = "<p>Hello <b>world</b>!</p>";
        assert_eq!(strip_html(s), "Hello world !");
    }

    #[test]
    fn html_entities() {
        let s = "&lt;p&gt;A &amp; B&nbsp;and &#65;";
        assert_eq!(strip_html(s), "<p>A & B and A");
    }

    #[test]
    fn html_script_dropped() {
        let s = "before<script>alert('x')</script>after";
        assert_eq!(strip_html(s), "before after");
    }

    #[test]
    fn md_heading() {
        let s = "# Title\nbody";
        let cleaned = strip_markdown(s);
        assert!(cleaned.contains("Title"));
        assert!(cleaned.contains("body"));
        assert!(!cleaned.contains("#"));
    }

    #[test]
    fn md_link() {
        let s = "see [the docs](https://x.com) for more";
        assert_eq!(strip_markdown(s), "see the docs for more");
    }

    #[test]
    fn md_code_block_dropped() {
        let s = "intro\n```\nlet x = 1;\n```\noutro";
        let cleaned = strip_markdown(s);
        assert!(cleaned.contains("intro"));
        assert!(cleaned.contains("outro"));
        assert!(!cleaned.contains("let x"));
    }

    #[test]
    fn md_emphasis() {
        let s = "this is *very* important and **bold**";
        let cleaned = strip_markdown(s);
        assert!(cleaned.contains("very"));
        assert!(cleaned.contains("bold"));
        assert!(!cleaned.contains("*"));
    }
}

#[cfg(test)]
mod properties {
    use super::*;
    use hegel::generators as gs;
    use hegel::TestCase;

    // Idempotence: stripping HTML twice gives the same result as stripping once.
    // Without this property a downstream re-strip (e.g. on a snippet that was
    // already cleaned at index time) would silently mangle the text further.
    #[hegel::test]
    fn strip_html_idempotent(tc: TestCase) {
        let s: String = tc.draw(gs::text());
        let once = strip_html(&s);
        let twice = strip_html(&once);
        assert_eq!(once, twice, "strip_html is not idempotent on input: {s:?}");
    }

    // Strip_html must never leave an opening tag in the output. The cleaner
    // collapses '<' followed by '>' into a single space; if we miss a case
    // the indexed term stream would inherit tag fragments.
    #[hegel::test]
    fn strip_html_removes_all_tags(tc: TestCase) {
        let s: String = tc.draw(gs::text());
        let cleaned = strip_html(&s);
        // We can't guarantee no raw '<' if the input was malformed (e.g. an
        // unclosed '<' at EOF). What we can guarantee: no full <...> pair.
        let mut in_tag = false;
        for c in cleaned.chars() {
            if c == '<' {
                in_tag = true;
            } else if c == '>' && in_tag {
                panic!("strip_html left a <...> pair in output: {cleaned:?} (input {s:?})");
            }
        }
    }

    // strip_markdown is also idempotent on its own output.
    #[hegel::test]
    fn strip_markdown_idempotent(tc: TestCase) {
        let s: String = tc.draw(gs::text());
        let once = strip_markdown(&s);
        let twice = strip_markdown(&once);
        assert_eq!(
            once, twice,
            "strip_markdown is not idempotent on input: {s:?}"
        );
    }

    // detect_and_clean either returns the input verbatim (no markup detected)
    // or returns the result of an idempotent stripper. In both cases applying
    // it twice should be a no-op.
    #[hegel::test]
    fn detect_and_clean_idempotent(tc: TestCase) {
        let s: String = tc.draw(gs::text());
        let once = detect_and_clean(&s);
        let twice = detect_and_clean(&once);
        assert_eq!(
            once, twice,
            "detect_and_clean is not idempotent on input: {s:?}"
        );
    }

    // Whitespace collapse never grows the string.
    #[hegel::test]
    fn collapse_ws_never_grows(tc: TestCase) {
        let s: String = tc.draw(gs::text());
        let out = collapse_ws(&s);
        assert!(
            out.len() <= s.len(),
            "collapse_ws grew the string: input {s:?} → output {out:?}",
        );
    }

    // Collapse_ws output has no run of 2+ consecutive ASCII whitespace chars.
    #[hegel::test]
    fn collapse_ws_no_double_space(tc: TestCase) {
        let s: String = tc.draw(gs::text());
        let out = collapse_ws(&s);
        let chars: Vec<char> = out.chars().collect();
        for w in chars.windows(2) {
            assert!(
                !(w[0].is_whitespace() && w[1].is_whitespace()),
                "double whitespace survived collapse: {out:?}",
            );
        }
    }
}
