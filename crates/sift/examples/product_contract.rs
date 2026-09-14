//! Check the CLI and HTTP contracts with a local model.

#[cfg(feature = "server")]
mod checks {
    use anyhow::{ensure, Context, Result};
    use serde_json::{json, Value};
    use std::fs::{self, File};
    use std::net::TcpListener;
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, Stdio};
    use std::thread::sleep;
    use std::time::{Duration, Instant};

    struct Process(Child);

    impl Drop for Process {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    struct Case {
        work: PathBuf,
        binary: PathBuf,
        model: String,
    }

    impl Drop for Case {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.work);
        }
    }

    impl Case {
        fn new(root: &Path, name: &str, binary: &Path, model: &str) -> Result<Self> {
            let work = root.join(format!("{name}-{}", std::process::id()));
            fs::create_dir(&work)?;
            let case = Self {
                work,
                binary: binary.to_owned(),
                model: model.into(),
            };
            let corpus = [
                json!({"id":"cat","text":"A domestic cat sleeps on the windowsill.",
                    "kind":"feline","price":2}),
                json!({"id":"kitten","text":"A playful kitten chases string.",
                    "kind":"feline","price":3}),
                json!({"id":"dog","text":"A dog plays in the park.","kind":"canine","price":4}),
            ];
            let jsonl = corpus
                .iter()
                .map(|row| format!("{row}\n"))
                .collect::<String>();
            fs::write(case.work.join("corpus.jsonl"), jsonl)?;
            Ok(case)
        }

        fn command(&self, arguments: &[&str]) -> Result<String> {
            let log = self.work.join("command.log");
            let file = File::create(&log)?;
            let mut process = Process(
                Command::new(&self.binary)
                    .args(arguments)
                    .stdout(file.try_clone()?)
                    .stderr(file)
                    .spawn()?,
            );
            let started = Instant::now();
            loop {
                if let Some(status) = process.0.try_wait()? {
                    let output = fs::read_to_string(log)?;
                    ensure!(status.success(), "Command failed: {arguments:?}\n{output}");
                    return Ok(output);
                }
                ensure!(
                    started.elapsed() < Duration::from_secs(120),
                    "Command exceeded its timeout."
                );
                sleep(Duration::from_millis(20));
            }
        }

        fn build(&self, operation: &str) -> Result<()> {
            self.command(&[
                operation,
                "--input",
                self.work.join("corpus.jsonl").to_str().unwrap(),
                if operation == "build" {
                    "--out"
                } else {
                    "--index"
                },
                self.work.join("docs.sift").to_str().unwrap(),
                "--model",
                &self.model,
                "--stop-df",
                "1.0",
                "--positions",
                "--rank-fields",
                "price",
            ])?;
            Ok(())
        }

        fn search(&self, query: &str) -> Result<String> {
            self.command(&[
                "search",
                self.work.join("docs.sift").to_str().unwrap(),
                query,
                "--semantic-weight",
                "0",
            ])
        }

        fn server(&self) -> Result<(Process, String)> {
            let listener = TcpListener::bind("127.0.0.1:0")?;
            let address = listener.local_addr()?.to_string();
            drop(listener);
            let log = File::create(self.work.join("server.log"))?;
            let mut process = Process(
                Command::new(&self.binary)
                    .args([
                        "serve",
                        "--artifacts",
                        self.work.to_str().unwrap(),
                        "--bind",
                        &address,
                    ])
                    .stdout(Stdio::from(log.try_clone()?))
                    .stderr(Stdio::from(log))
                    .spawn()?,
            );
            let base = format!("http://{address}");
            for _ in 0..100 {
                ensure!(
                    process.0.try_wait()?.is_none(),
                    "The test server exited during startup."
                );
                if ureq::get(&format!("{base}/readyz"))
                    .timeout(Duration::from_millis(200))
                    .call()
                    .is_ok()
                {
                    return Ok((process, base));
                }
                sleep(Duration::from_millis(50));
            }
            anyhow::bail!("The test server exceeded its startup budget.")
        }
    }

    fn post(base: &str, endpoint: &str, body: Value) -> Result<Value> {
        Ok(ureq::post(&format!("{base}{endpoint}"))
            .timeout(Duration::from_secs(60))
            .send_json(body)?
            .into_json()?)
    }

    fn query(text: &str) -> Value {
        json!({"index":"docs", "q":text, "blend_alpha":0, "cache":false})
    }

    fn single_segment(case: &Case) -> Result<()> {
        case.build("build")?;
        assert!(case.search("cat")?.contains("# 1 hits"));
        let (_process, base) = case.server()?;
        let mut request = query("cat");
        request["highlight"] = json!(true);
        let response = post(&base, "/search", request)?;
        assert_eq!(response["hits"][0]["doc_id"], "cat");
        assert_eq!(response["total"], 1);
        assert_eq!(
            response["hits"][0]["payload"],
            json!({"id":"cat",
            "text":"A domestic cat sleeps on the windowsill.","kind":"feline","price":2})
        );
        assert!(response["hits"][0]["snippet_html"]
            .as_str()
            .unwrap()
            .contains("<mark>cat</mark>"));
        assert!(response.get("latency_us").is_some());
        assert_eq!(
            post(&base, "/search", query("\"domestic cat\""))?["hits"][0]["doc_id"],
            "cat"
        );
        let mut filtered = query("cat");
        filtered["filter"] = json!([{"field":"kind","eq":"canine"}]);
        assert_eq!(post(&base, "/search", filtered)?["hits"], json!([]));
        let mut page = query("cat");
        page["offset"] = json!(1);
        let page = post(&base, "/search", page)?;
        assert_eq!(page["hits"], json!([]));
        assert_eq!(page["total"], 1);
        let mut facets = query("cat");
        facets["facets"] = json!(["price"]);
        assert_eq!(
            post(&base, "/search", facets)?["facets"]["price"],
            json!([{"value":2.0,"count":1},{"value":3.0,"count":1}])
        );
        for (endpoint, invalid) in [
            ("/search", json!({"index":"docs","q":"cat","blend_alpha":2})),
            (
                "/add",
                json!({"index":"../escape","docs":[{"id":"x","text":"cat"}]}),
            ),
            (
                "/add",
                json!({"index":"docs","docs":[{"id":true,"text":"cat"}]}),
            ),
            (
                "/add",
                json!({"index":"docs","docs":[{"id":"a\nb","text":"cat"}]}),
            ),
        ] {
            let response = ureq::post(&format!("{base}{endpoint}"))
                .timeout(Duration::from_secs(60))
                .send_json(invalid);
            assert!(matches!(response, Err(ureq::Error::Status(400, _))));
        }
        Ok(())
    }

    fn cli_writes(case: &Case) -> Result<()> {
        case.build("add")?;
        let index = case.work.join("docs.sift");
        case.command(&["delete", "--index", index.to_str().unwrap(), "--id", "cat"])?;
        assert!(case.search("cat")?.contains("# 0 hits"));
        case.command(&["compact", "--index", index.to_str().unwrap()])?;
        assert!(case.search("cat")?.contains("# 0 hits"));
        assert!(case.search("dog")?.contains("# 1 hits"));
        Ok(())
    }

    fn http_writes(case: &Case) -> Result<()> {
        case.build("add")?;
        {
            let (_process, base) = case.server()?;
            let added = post(
                &base,
                "/add",
                json!({"index":"docs",
                "docs":[{"id":"bird","text":"A sparrow flies over a tree."}]}),
            )?;
            assert_eq!(added["affected"], 1);
            assert_eq!(
                post(&base, "/search", query("sparrow"))?["hits"][0]["doc_id"],
                "bird"
            );
            post(
                &base,
                "/add",
                json!({"index":"docs","upsert":true,
                "docs":[{"id":"bird","text":"An eagle flies over a tree."}]}),
            )?;
            assert_eq!(post(&base, "/search", query("sparrow"))?["hits"], json!([]));
            assert_eq!(
                post(&base, "/delete", json!({"index":"docs","ids":["dog"]}))?["affected"],
                1
            );
        }
        let (_process, base) = case.server()?;
        assert_eq!(post(&base, "/search", query("dog"))?["hits"], json!([]));
        assert_eq!(
            post(&base, "/search", query("eagle"))?["hits"][0]["doc_id"],
            "bird"
        );
        Ok(())
    }

    pub fn run() -> Result<()> {
        let model = std::env::args()
            .nth(1)
            .context("Provide a local model directory.")?;
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/product-contract");
        fs::create_dir_all(&root)?;
        let binary = std::env::var_os("SIFT_BIN")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                std::env::current_exe()
                    .unwrap()
                    .parent()
                    .unwrap()
                    .parent()
                    .unwrap()
                    .join("sift")
            });
        ensure!(binary.is_file(), "Build the Sift binary or set SIFT_BIN.");
        single_segment(&Case::new(&root, "single", &binary, &model)?)?;
        cli_writes(&Case::new(&root, "cli", &binary, &model)?)?;
        http_writes(&Case::new(&root, "http", &binary, &model)?)?;
        println!("CLI search, HTTP contracts, writes, compaction, and reopen checks passed.");
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    #[cfg(feature = "server")]
    return checks::run();
    #[cfg(not(feature = "server"))]
    anyhow::bail!("Enable the server feature for this check.");
}
