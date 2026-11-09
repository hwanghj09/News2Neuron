from __future__ import annotations

import gzip
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser


class FetchFailure(Exception):
    def __init__(self, reason, status=None):
        super().__init__(reason)
        self.reason, self.status = reason, status


class _Redirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, hosts):
        self.hosts = set(hosts)

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urlsplit(newurl)
        if target.scheme != "https" or target.hostname not in self.hosts:
            raise FetchFailure("redirect_outside_publisher_allowlist", code)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class CachedHTTP:
    """Public-only requests, truthful UA, per-host pacing and immutable responses."""
    def __init__(self, config, root):
        self.config, self.root = config, Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hosts = {host for p in config["publishers"] for host in p["allowed_hosts"]}
        self.opener = urllib.request.build_opener(_Redirect(self.hosts))
        self.host_locks = {h: threading.Lock() for h in self.hosts}
        self.last = {}
        self.robots = {}
        self.log_lock = threading.Lock()

    def _log(self, data):
        with self.log_lock, (self.root/"requests.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False)+"\n")

    def _request(self, url, max_bytes):
        host = urlsplit(url).hostname
        if urlsplit(url).scheme != "https" or host not in self.hosts:
            raise FetchFailure("url_outside_publisher_allowlist")
        key = hashlib.sha256(url.encode()).hexdigest()
        receipt = self.root/"receipts"/(key+".json")
        if receipt.exists():
            meta = json.loads(receipt.read_text())
            if meta["status"] == 200:
                payload = gzip.decompress((self.root/"responses"/(meta["sha256"]+".gz")).read_bytes())
                if hashlib.sha256(payload).hexdigest() != meta["sha256"]:
                    raise FetchFailure("cached_response_hash_mismatch")
                return payload, meta
            raise FetchFailure(meta["reason"], meta["status"])
        with self.host_locks[host]:
            delay = max(self.config["request_interval_seconds"], self.robots.get(host, (None,0))[1])
            gap = delay-(time.monotonic()-self.last.get(host,0))
            if gap > 0:
                time.sleep(gap)
            self.last[host] = time.monotonic()
            meta = {"url":url,"retrieved_at":datetime.now(timezone.utc).isoformat(),"user_agent":self.config["user_agent"]}
            try:
                request = urllib.request.Request(url,headers={"User-Agent":self.config["user_agent"],"Accept-Encoding":"identity"})
                with self.opener.open(request,timeout=self.config["request_timeout_seconds"]) as response:
                    payload = response.read(max_bytes+1)
                    if len(payload)>max_bytes:
                        raise FetchFailure("response_size_limit",response.status)
                    meta.update(status=response.status,final_url=response.geturl(),content_type=response.headers.get("Content-Type",""),
                                bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
                if meta["status"]!=200:
                    raise FetchFailure("non_success_http",meta["status"])
                (self.root/"responses").mkdir(exist_ok=True)
                (self.root/"responses"/(meta["sha256"]+".gz")).write_bytes(gzip.compress(payload,mtime=0))
            except urllib.error.HTTPError as e:
                meta.update(status=e.code,reason=f"http_{e.code}")
            except FetchFailure as e:
                meta.update(status=0,http_status=e.status,reason=e.reason)
            except Exception as e:
                meta.update(status=0,reason=type(e).__name__)
            receipt.parent.mkdir(exist_ok=True)
            receipt.write_text(json.dumps(meta,ensure_ascii=False),encoding="utf-8")
            self._log(meta)
            if meta["status"]!=200:
                raise FetchFailure(meta["reason"],meta["status"])
            return payload,meta

    def policy(self, host):
        if host not in self.robots:
            url = "https://"+host+"/robots.txt"
            parser = RobotFileParser()
            try:
                raw,meta = self._request(url,500_000)
                parser.parse(raw.decode("utf-8",errors="replace").splitlines())
                delay = parser.crawl_delay(self.config["user_agent"]) or 0
                self.robots[host] = (parser,delay)
            except FetchFailure as e:
                if e.status==404:
                    parser.parse(["User-agent: *","Allow: /"])
                    self.robots[host]=(parser,0)
                else:
                    parser.parse(["User-agent: *","Disallow: /"])
                    self.robots[host]=(parser,0)
        return self.robots[host][0]

    def fetch(self,url,xml=False):
        host = urlsplit(url).hostname
        if host not in self.hosts:
            raise FetchFailure("url_outside_publisher_allowlist")
        if not self.policy(host).can_fetch(self.config["user_agent"],url):
            self._log({"url":url,"status":0,"reason":"robots_denied"})
            raise FetchFailure("robots_denied")
        return self._request(url,self.config["max_xml_bytes"] if xml else self.config["max_response_bytes"])
