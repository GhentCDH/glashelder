use std::{
    num::NonZeroU32,
    sync::{
        Arc, LazyLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::Context as _;
use governor::{DefaultKeyedRateLimiter, Quota};
use reqwest::{
    Client, ClientBuilder, Request, RequestBuilder, Response, StatusCode,
    header::{ACCEPT, HeaderMap, HeaderName, HeaderValue, RANGE, USER_AGENT as UA},
};
use serde::{Serialize, de::DeserializeOwned};
use tokio::sync::Semaphore;

pub mod image;
pub mod inventory;

const USER_AGENT: &str = "Ghent Centre for Digital Humanities <sysadmin.ghentcdh@ugent.be>";

static REQUESTS: AtomicU64 = AtomicU64::new(0);
static START: LazyLock<Instant> = LazyLock::new(Instant::now);

/// A request that takes longer fails, and is tried again.
const TIMEOUT: Duration = Duration::from_secs(60);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Tries of one request, with a wait of 2, 4, 8, ... seconds in between.
const ATTEMPTS: u32 = 5;

/// Whether trying again can help: a timeout, a dropped connection, a server
/// error or too many requests. A 404 or a body that is not JSON will not change.
fn retryable(error: &anyhow::Error) -> bool {
    error
        .chain()
        .filter_map(|cause| cause.downcast_ref::<reqwest::Error>())
        .any(|e| {
            e.is_timeout()
                || e.is_connect()
                || e.is_request()
                || e.is_body()
                || e.status().is_some_and(|status| {
                    status.is_server_error() || status == StatusCode::TOO_MANY_REQUESTS
                })
        })
}

pub struct UnstirClientConfig {
    concurrency: usize,
    per_second: NonZeroU32,
}

impl Default for UnstirClientConfig {
    fn default() -> Self {
        Self {
            concurrency: 2,
            per_second: NonZeroU32::new(6).expect("6 is nonzero"),
        }
    }
}

#[must_use]
pub struct UnstirRequest<'a> {
    client: &'a UnstirClient,
    inner: RequestBuilder,
}

impl<'a> UnstirRequest<'a> {
    pub fn query<T: Serialize + ?Sized>(self, query: &T) -> Self {
        Self {
            inner: self.inner.query(query),
            ..self
        }
    }

    pub fn header<K, V>(self, key: K, value: V) -> Self
    where
        HeaderName: TryFrom<K>,
        <HeaderName as TryFrom<K>>::Error: Into<http::Error>,
        HeaderValue: TryFrom<V>,
        <HeaderValue as TryFrom<V>>::Error: Into<http::Error>,
    {
        Self {
            inner: self.inner.header(key, value),
            ..self
        }
    }

    pub async fn json<T: DeserializeOwned>(self) -> anyhow::Result<T> {
        let client = self.client;
        client
            .limited(self.inner, |r| async { Ok(r.json().await?) })
            .await
    }

    pub async fn bytes(self) -> anyhow::Result<bytes::Bytes> {
        let client = self.client;
        client
            .limited(self.inner, |r| async { Ok(r.bytes().await?) })
            .await
    }
}

/// Reqwest-backed managed http client with rate limiting and concurrency limits.
///
/// Automatically sets `Accept: application/json` and `User-Agent` headings
#[derive(Clone)]
pub struct UnstirClient {
    client: Client,
    permits: Arc<Semaphore>,
    rate: Arc<DefaultKeyedRateLimiter<String>>,
}

impl<'a> UnstirClient {
    pub fn new(config: UnstirClientConfig) -> anyhow::Result<Self> {
        let mut headers = HeaderMap::new();
        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
        headers.insert(UA, HeaderValue::from_static(USER_AGENT));

        let client = ClientBuilder::new()
            .default_headers(headers)
            .timeout(TIMEOUT)
            .connect_timeout(CONNECT_TIMEOUT)
            .build()?;

        Ok(Self {
            client,
            permits: Arc::new(Semaphore::new(config.concurrency)),
            rate: Arc::new(DefaultKeyedRateLimiter::keyed(Quota::per_second(
                config.per_second,
            ))),
        })
    }

    pub fn get(&'a self, url: impl reqwest::IntoUrl) -> UnstirRequest<'a> {
        UnstirRequest {
            client: self,
            inner: self.client.get(url),
        }
    }

    /// `req`, read with `read`, tried again on errors that can pass (see
    /// [`retryable`]). The permit is given back while waiting to try again.
    async fn limited<T, F, Fut>(&self, req: RequestBuilder, read: F) -> anyhow::Result<T>
    where
        F: Fn(Response) -> Fut,
        Fut: Future<Output = anyhow::Result<T>>,
    {
        let req = req.build()?;
        let count = REQUESTS.fetch_add(1, Ordering::Relaxed);
        let mut attempt = 1;
        loop {
            let this = req.try_clone().context("a request with a stream body")?;
            match self.attempt(this, count, &read).await {
                Err(error) if attempt < ATTEMPTS && retryable(&error) => {
                    let wait = Duration::from_secs(2u64.pow(attempt));
                    eprintln!(
                        "\tretrying request {count} in {wait:?}, attempt {} of {ATTEMPTS}: {error:#}",
                        attempt + 1
                    );
                    tokio::time::sleep(wait).await;
                    attempt += 1;
                }
                result => return result.with_context(|| format!("GET {}", req.url())),
            }
        }
    }

    async fn attempt<T, F, Fut>(&self, req: Request, count: u64, read: &F) -> anyhow::Result<T>
    where
        F: Fn(Response) -> Fut,
        Fut: Future<Output = anyhow::Result<T>>,
    {
        let host = req.url().host_str().unwrap_or_default().to_owned();
        let _permit = self.permits.acquire().await?;
        self.rate.until_key_ready(&host).await;
        let range = req
            .headers()
            .get(RANGE)
            .and_then(|range| range.to_str().ok())
            .map(|range| format!(" ({range})"))
            .unwrap_or_default();
        eprintln!(
            "starting request {count} at {:.3?} {}{range}",
            START.elapsed(),
            req.url()
        );
        let result = match self.client.execute(req).await {
            Ok(response) => match response.error_for_status() {
                Ok(response) => read(response).await,
                Err(error) => Err(error.into()),
            },
            Err(error) => Err(error.into()),
        };
        let outcome = if result.is_ok() { "finished" } else { "failed" };
        eprintln!(
            "\t{outcome} request {count} at {:.3?} ({host})",
            START.elapsed()
        );
        result
    }
}
