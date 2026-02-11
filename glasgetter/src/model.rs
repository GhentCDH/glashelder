use std::fmt::Display;

use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct Artist {
    pub name: String,
    pub resource: String,
    pub artwork_count: u64,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Artwork {
    pub name: String,
    pub resource: String,
    pub artist_name: String,
    pub artist_resource: String,
}

impl Display for Artist {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} @ {}", self.name, self.resource)
    }
}

impl Display for Artwork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} @ {}", self.name, self.resource)
    }
}
