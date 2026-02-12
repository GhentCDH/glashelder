use std::time::Duration;

use reqwest::{Client, Response, StatusCode};

use crate::error::{Error, Result};

const BASE_URL: &str = "https://www.wikiart.org";
const MAX_RETRIES: u32 = 3;
const INITIAL_BACKOFF_MS: u64 = 1000;

/// Build a configured HTTP client
pub fn build_client() -> Result<Client> {
    Client::builder()
        .timeout(Duration::from_secs(30))
        .user_agent("glasgetter/0.1.0 (artwork research tool)")
        .build()
        .map_err(Error::Http)
}

/// Construct a full WikiArt URL from a resource path
#[inline]
pub fn wiki_url(resource: &str) -> String {
    // The resource possibly isn't a relative path, if it refers to an
    // external resource
    if resource.contains("http:") {
        return String::from(resource);
    }

    match resource.chars().next().is_some_and(|c| c == '/') {
        true => format!("{BASE_URL}{resource}"),
        false => format!("{BASE_URL}/{resource}"),
    }
}

/// Fetch a URL with automatic retry on transient failures
pub async fn fetch_with_retry(client: &Client, url: &str) -> Result<Response> {
    let mut last_error = None;
    let mut backoff_ms = INITIAL_BACKOFF_MS;

    for attempt in 0..MAX_RETRIES {
        if attempt > 0 {
            tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
            backoff_ms *= 2; // exponential backoff
        }

        match client.get(url).send().await {
            Ok(response) => {
                let status = response.status();

                if status.is_success() {
                    return Ok(response);
                }

                if status == StatusCode::TOO_MANY_REQUESTS {
                    let retry_after = response
                        .headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|s| s.parse().ok());

                    if let Some(secs) = retry_after {
                        // wait the requested time before retrying
                        tokio::time::sleep(Duration::from_secs(secs)).await;
                    }

                    last_error = Some(Error::RateLimited { retry_after });
                    continue;
                }

                if status.is_server_error() {
                    last_error = Some(Error::Server {
                        status: status.as_u16(),
                        url: url.to_string(),
                    });
                    continue;
                }

                // client errors (4xx except 429) are not retryable
                return Err(Error::Server {
                    status: status.as_u16(),
                    url: url.to_string(),
                });
            }
            Err(e) => {
                if e.is_timeout() || e.is_connect() {
                    last_error = Some(Error::Http(e));
                    continue;
                }
                return Err(Error::Http(e));
            }
        }
    }

    Err(last_error.unwrap_or(Error::Server {
        status: 0,
        url: format!("{url} (max retries exceeded)"),
    }))
}

/// Fetch HTML content from a URL with retry
pub async fn fetch_html(client: &Client, url: &str) -> Result<String> {
    let response = fetch_with_retry(client, url).await?;
    response.text().await.map_err(Error::Http)
}

/// Fetch binary content (for images) with retry
pub async fn fetch_bytes(client: &Client, url: &str) -> Result<bytes::Bytes> {
    let response = fetch_with_retry(client, url).await?;
    response.bytes().await.map_err(Error::Http)
}
