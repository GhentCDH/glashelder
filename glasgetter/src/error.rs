use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// HTTP request failed
    Http(reqwest::Error),
    /// Rate limited by server (HTTP 429)
    RateLimited { retry_after: Option<u64> },
    /// Server error (5xx)
    Server { status: u16, url: String },
    /// Unexpected HTML structure - selector didn't match
    UnexpectedHtml { context: String, url: String },
    /// File I/O error
    Io(std::io::Error),
    /// JSON parsing error
    Json(serde_json::Error),
    /// File access is restricted in Belgium
    Restricted,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Http(e) => write!(f, "http error: {e}"),
            Error::RateLimited { retry_after } => {
                write!(f, "rate limited by server")?;
                if let Some(secs) = retry_after {
                    write!(f, " (retry after {secs}s)")?;
                }
                Ok(())
            }
            Error::Server { status, url } => {
                write!(f, "server error {status} for {url}")
            }
            Error::UnexpectedHtml { context, url } => {
                write!(f, "unexpected html structure: {context} (url: {url})")
            }
            Error::Io(e) => write!(f, "i/o error: {e}"),
            Error::Json(e) => write!(f, "json error: {e}"),
            Error::Restricted => write!(f, "restricted access"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Http(e) => Some(e),
            Error::Io(e) => Some(e),
            Error::Json(e) => Some(e),
            _ => None,
        }
    }
}

impl From<reqwest::Error> for Error {
    fn from(e: reqwest::Error) -> Self {
        Error::Http(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Json(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
