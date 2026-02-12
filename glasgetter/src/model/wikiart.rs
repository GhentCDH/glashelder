use std::fmt;

use serde::{Deserialize, Serialize};

use super::common::{Creator, Source, UnifiedArtwork};

/// WikiArt artist information
#[derive(Clone, Debug)]
pub struct WikiArtist {
    pub name: String,
    pub resource: String,
    pub artwork_count: u64,
}

impl fmt::Display for WikiArtist {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} @ {}", self.name, self.resource)
    }
}

/// WikiArt artwork information (internal representation)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WikiArtwork {
    pub name: String,
    pub resource: String,
    pub artist_name: String,
    pub artist_resource: String,
}

impl fmt::Display for WikiArtwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} @ {}", self.name, self.resource)
    }
}

impl From<WikiArtwork> for UnifiedArtwork {
    fn from(w: WikiArtwork) -> Self {
        UnifiedArtwork {
            id: w.resource.clone(),
            title: w.name,
            resource: w.resource,
            source: Source::WikiArt,
            creator: Some(Creator {
                name: w.artist_name,
                resource: Some(w.artist_resource),
            }),
            image_url: None, // WikiArt requires a separate fetch
        }
    }
}
