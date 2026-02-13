use std::fmt;

use serde::{Deserialize, Serialize};

/// The source of an artwork
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    WikiArt,
    Balat,
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Source::WikiArt => write!(f, "wikiart"),
            Source::Balat => write!(f, "balat"),
        }
    }
}

/// Creator/artist information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Creator {
    pub name: String,
    pub resource: Option<String>,
}

/// Unified artwork representation across all sources
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnifiedArtwork {
    /// Unique identifier within the source
    pub id: String,
    /// Title of the artwork
    pub title: String,
    /// Resource path or URL to the artwork page
    pub resource: String,
    /// Which source this artwork came from
    pub source: Source,
    /// Creator/artist information
    pub creator: Option<Creator>,
    /// Direct image URL (BALAT has this upfront, WikiArt fetches lazily)
    pub image_url: Option<String>,
}

impl fmt::Display for UnifiedArtwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {} @ {}", self.source, self.title, self.resource)
    }
}
