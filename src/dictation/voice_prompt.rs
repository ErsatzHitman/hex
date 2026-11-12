//! Labs › Voice prompt: a dictation that contains a spoken trigger phrase
//! (`labs.voice_prompt_phrase`, "rewrite this" by default) is not pasted as
//! said. Everything spoken after the phrase is a prompt for the AI model, and
//! everything before it is the text that prompt applies to:
//!
//! ```text
//! "We ship Friday. Rewrite this, make it formal."
//!  └── text ─────┘ └ phrase ──┘ └── prompt ───┘
//! ```
//!
//! Without a prompt ("… rewrite this.") the text is rewritten clearly and
//! concisely; without text ("Rewrite this: write a haiku about rain") the
//! prompt alone asks the model for what to paste.
//!
//! Pure, with no I/O. Matching is token aligned and ignores case,
//! punctuation, whitespace, hyphens, and apostrophes, with no edit-distance
//! fuzz: a false match replaces what the person said, while a missed one
//! pastes the phrase where it is seen and undone. When the phrase is said
//! more than once, the last time with words after it (or at the very end)
//! splits the dictation.

/// A usable trigger phrase, squashed to lowercase letters and digits.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Trigger {
    squashed: String,
}

/// The default trigger phrase.
pub const DEFAULT_PHRASE: &str = "rewrite this";
/// Longest phrase accepted, in characters.
pub const MAX_PHRASE_CHARS: usize = 80;
/// Fewest words a phrase needs: one common word would hijack dictations.
pub const MIN_WORDS: usize = 2;
/// Fewest letters a phrase needs (a CJK character counts as two).
pub const MIN_WEIGHT: usize = 6;

impl Trigger {
    /// The trigger for `phrase`, or `None` when it has fewer than two words,
    /// fewer than six letters, or more than 80 characters.
    pub fn new(phrase: &str) -> Option<Self> {
        let phrase = phrase.trim();
        if phrase.chars().count() > MAX_PHRASE_CHARS {
            return None;
        }
        let words = tokens(phrase);
        let squashed: String = words.iter().map(|word| word.squashed.as_str()).collect();
        let weight: usize = squashed
            .chars()
            .map(|c| if is_cjk(c) { 2 } else { 1 })
            .sum();
        (words.len() >= MIN_WORDS && weight >= MIN_WEIGHT).then_some(Self { squashed })
    }
}

/// What [`detect`] found in a dictation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Detection {
    /// The phrase isn't there: process the dictation as usual.
    NoMatch,
    /// Only the phrase (and perhaps politeness): nothing to paste.
    OnlyPhrase,
    /// The phrase split the dictation. `text` is what came before it
    /// (empty when the dictation began with it); `prompt` is what came after
    /// (`None` when the phrase ended it).
    Prompt {
        text: String,
        prompt: Option<String>,
    },
}

/// Words dropped from the end of the text, before the phrase ("…, and
/// rewrite this").
const CONNECTIVES: &[&str] = &["and", "then", "so", "please", "also", "now"];
/// Words dropped from the start of the prompt ("… rewrite this, and then
/// make it formal").
const LEADING_CONNECTIVES: &[&str] = &["and", "then", "also", "so", "but", "just", "to"];
/// Words that alone are only politeness after the phrase ("… rewrite this,
/// thank you"), not a prompt.
const POLITE: &[&str] = &[
    "please", "thanks", "thank", "you", "very", "much", "so", "a", "lot",
];
/// Characters trimmed from the end of the text once the phrase is cut off:
/// the clause break that led into it. Sentence-final `. ! ?` stay, as do
/// closing quotes and brackets; opening ones go, so a phrase a recognizer
/// quoted leaves nothing dangling.
const TRAILING_SEPARATORS: &[char] = &[
    ',', ';', ':', '-', '\u{2013}', '\u{2014}', '\u{2026}', '\u{FF0C}', '\u{3001}', '\u{FF1B}',
    '\u{FF1A}', '"', '\u{201C}', '\u{201E}', '\u{00AB}', '\u{2039}', '(', '[', '{', '\u{300C}',
    '\u{300E}',
];

fn is_separator(c: char) -> bool {
    c.is_whitespace() || TRAILING_SEPARATORS.contains(&c)
}

fn is_sentence_mark(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '\u{3002}' | '\u{FF01}' | '\u{FF1F}')
}

/// Finds `trigger` in `transcript`.
pub fn detect(transcript: &str, trigger: &Trigger) -> Detection {
    let words = tokens(transcript);
    if words.is_empty() || trigger.squashed.is_empty() {
        return Detection::NoMatch;
    }
    // The last occurrence wins: the prompt after it is short, while the text
    // before it may say the phrase in passing.
    for end in (1..=words.len()).rev() {
        if let Some(start) = align(&words, end, &trigger.squashed) {
            let text = trim_text(&transcript[..words[start].start]);
            let prompt = (end < words.len())
                .then(|| trim_prompt(&transcript[words[end - 1].end..]))
                .flatten();
            return match (text.is_empty(), prompt) {
                (true, None) => Detection::OnlyPhrase,
                (_, prompt) => Detection::Prompt { text, prompt },
            };
        }
    }
    Detection::NoMatch
}

/// The words after the phrase without the separators and connectives that
/// led into them; `None` when nothing but politeness is left.
fn trim_prompt(raw: &str) -> Option<String> {
    let lead = |c: char| is_separator(c) || is_sentence_mark(c);
    let mut rest = raw.trim_start_matches(lead);
    while let Some(first) = tokens(rest)
        .first()
        .filter(|word| word.start == 0 && LEADING_CONNECTIVES.contains(&word.squashed.as_str()))
    {
        rest = rest[first.end..].trim_start_matches(lead);
    }
    let words = tokens(rest);
    if words
        .iter()
        .all(|word| POLITE.contains(&word.squashed.as_str()))
    {
        return None;
    }
    Some(rest.trim_end().to_owned())
}

/// The text before the phrase, trimmed, without the connectives that led
/// into it ("… Friday, and then rewrite this"); empty when no word is left.
fn trim_text(raw: &str) -> String {
    let mut end = raw.trim_end_matches(is_separator).len();
    let words = tokens(&raw[..end]);
    let mut kept = words.len();
    // A connective only counts as the last thing before the phrase: a
    // sentence that ends "… and so." keeps its word.
    while kept > 0
        && words[kept - 1].end == end
        && CONNECTIVES.contains(&words[kept - 1].squashed.as_str())
    {
        end = raw[..words[kept - 1].start]
            .trim_end_matches(is_separator)
            .len();
        kept -= 1;
    }
    if kept == 0 {
        return String::new();
    }
    raw[..end].trim().to_owned()
}

/// The index of the first word of `squashed` when it ends exactly at word
/// `end`, word aligned.
fn align(words: &[Word], end: usize, squashed: &str) -> Option<usize> {
    let mut joined = String::new();
    for index in (0..end).rev() {
        joined.insert_str(0, &words[index].squashed);
        if joined.len() == squashed.len() {
            return (joined == squashed).then_some(index);
        }
        if joined.len() > squashed.len() {
            return None;
        }
    }
    None
}

/// One word: its lowercase letters and digits (apostrophes dropped) and its
/// byte span in the text it came from.
#[derive(Clone, Debug, Eq, PartialEq)]
struct Word {
    squashed: String,
    start: usize,
    end: usize,
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3040..=0x30FF | 0x31F0..=0x31FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF
        | 0x20000..=0x2FFFF)
}

/// Combining marks and joiners that continue a word.
fn is_mark(c: char) -> bool {
    matches!(c as u32,
        0x0300..=0x036F | 0x0483..=0x0489 | 0x0591..=0x05C7 | 0x064B..=0x065F | 0x0670
        | 0x0900..=0x0DFF | 0x0E31 | 0x0E34..=0x0E3A | 0x0E47..=0x0E4E | 0x200C | 0x200D
        | 0x20D0..=0x20FF | 0xFE00..=0xFE0F)
}

/// Fullwidth ASCII variants and the ideographic space, folded.
fn fold(c: char) -> char {
    match c as u32 {
        0xFF01..=0xFF5E => char::from_u32(c as u32 - 0xFEE0).unwrap_or(c),
        0x3000 => ' ',
        _ => c,
    }
}

fn tokens(text: &str) -> Vec<Word> {
    let mut words = Vec::new();
    let mut current: Option<Word> = None;
    for (index, original) in text.char_indices() {
        let c = fold(original);
        let end = index + original.len_utf8();
        // Glued to the word: "don't" is "dont".
        if matches!(c, '\'' | '\u{2019}' | '\u{2018}' | '`' | '\u{02BC}') {
            continue;
        }
        if is_cjk(c) {
            words.extend(current.take());
            words.push(Word {
                squashed: c.to_string(),
                start: index,
                end,
            });
        } else if c.is_alphanumeric() || (is_mark(c) && current.is_some()) {
            let word = current.get_or_insert_with(|| Word {
                squashed: String::new(),
                start: index,
                end,
            });
            word.squashed.extend(c.to_lowercase());
            word.end = end;
        } else {
            words.extend(current.take());
        }
    }
    words.extend(current);
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detect_default(text: &str) -> Detection {
        detect(text, &Trigger::new(DEFAULT_PHRASE).unwrap())
    }

    fn prompt(text: &str, prompt: &str) -> Detection {
        Detection::Prompt {
            text: text.into(),
            prompt: Some(prompt.into()),
        }
    }

    fn no_prompt(text: &str) -> Detection {
        Detection::Prompt {
            text: text.into(),
            prompt: None,
        }
    }

    #[test]
    fn everything_after_the_phrase_is_the_prompt() {
        assert_eq!(
            detect_default("We ship Friday. Rewrite this, make it formal."),
            prompt("We ship Friday.", "make it formal.")
        );
        assert_eq!(
            detect_default("We ship Friday, rewrite this and then translate it to Spanish"),
            prompt("We ship Friday", "translate it to Spanish")
        );
        assert_eq!(
            detect_default("Send the deck tomorrow. Rewrite this. As three bullet points, please."),
            prompt("Send the deck tomorrow.", "As three bullet points, please.")
        );
    }

    #[test]
    fn a_phrase_that_ends_the_dictation_has_no_prompt() {
        assert_eq!(
            detect_default("Send me the report by Friday. Rewrite this."),
            no_prompt("Send me the report by Friday.")
        );
        assert_eq!(
            detect_default("send me the report by friday rewrite this"),
            no_prompt("send me the report by friday")
        );
        assert_eq!(
            detect_default("Send it Friday, and then rewrite this!"),
            no_prompt("Send it Friday")
        );
    }

    #[test]
    fn politeness_after_the_phrase_is_not_a_prompt() {
        assert_eq!(
            detect_default("We ship Friday. Rewrite this, please."),
            no_prompt("We ship Friday.")
        );
        assert_eq!(
            detect_default("We ship Friday. Rewrite this. Thank you very much."),
            no_prompt("We ship Friday.")
        );
    }

    #[test]
    fn a_prompt_without_text_asks_for_new_text() {
        assert_eq!(
            detect_default("Rewrite this: write a haiku about rain."),
            prompt("", "write a haiku about rain.")
        );
        assert_eq!(
            detect_default("Please rewrite this as a thank-you note to Sam"),
            prompt("", "as a thank-you note to Sam")
        );
    }

    #[test]
    fn only_the_phrase_leaves_nothing() {
        assert_eq!(detect_default("Rewrite this."), Detection::OnlyPhrase);
        assert_eq!(
            detect_default("REWRITE THIS, thanks"),
            Detection::OnlyPhrase
        );
        assert_eq!(detect_default("And rewrite this"), Detection::OnlyPhrase);
    }

    #[test]
    fn spelling_and_spacing_do_not_matter_but_words_do() {
        assert_eq!(
            detect_default("I need it Monday.  Re-write   this\nin French"),
            prompt("I need it Monday.", "in French")
        );
        assert_eq!(detect_default("Please overwrite this."), Detection::NoMatch);
        assert_eq!(
            detect_default("Please write this down."),
            Detection::NoMatch
        );
        assert_eq!(detect_default("Rewrite these lines."), Detection::NoMatch);
        assert_eq!(detect_default("We rewriter this."), Detection::NoMatch);
        assert_eq!(detect_default(""), Detection::NoMatch);
    }

    #[test]
    fn the_last_phrase_splits_the_dictation() {
        assert_eq!(
            detect_default("I said rewrite this once. Rewrite this, shorter"),
            prompt("I said rewrite this once.", "shorter")
        );
    }

    #[test]
    fn a_quoted_phrase_leaves_no_dangling_quote() {
        assert_eq!(
            detect_default("Call me at noon \u{201C}rewrite this\u{201D}"),
            no_prompt("Call me at noon")
        );
    }

    #[test]
    fn custom_phrases_and_their_limits() {
        let trigger = Trigger::new("  Hey Hex, Make It ").unwrap();
        assert_eq!(
            detect("Lunch at one. Hey hex make it a question", &trigger),
            prompt("Lunch at one.", "a question")
        );
        assert_eq!(Trigger::new("rewrite"), None, "one word");
        assert_eq!(Trigger::new("do it"), None, "too few letters");
        assert_eq!(Trigger::new(""), None);
        assert_eq!(Trigger::new(&"word ".repeat(20)), None, "too long");
        assert!(Trigger::new("改写这个").is_some(), "CJK counts double");
        assert_eq!(
            detect(
                "明天开会改写这个更正式",
                &Trigger::new("改写 这个").unwrap()
            ),
            prompt("明天开会", "更正式")
        );
    }
}
