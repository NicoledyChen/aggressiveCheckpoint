#!/usr/bin/env python3
"""
DeepInfra deployment helper (serverless/custom LLM).

Uses DeepInfra's public OpenAPI (https://api.deepinfra.com/docs) endpoints:
- POST /deploy/llm                    (create LLM deployment)
- GET  /deploy/{deploy_id}            (status)
- PUT  /deploy/{deploy_id}            (scale settings)
- DELETE /deploy/{deploy_id}          (delete)
- GET  /deploy/llm/gpu_availability   (choose a GPU)

Auth:
- Bearer token via DEEPINFRA_API_KEY (recommended) or OPENAI_API_KEY.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx


DEFAULT_DEPLOY_BASE_URL = "https://api.deepinfra.com"


@dataclasses.dataclass(frozen=True)
class GPUOption:
    gpu: str
    num_gpus: int
    gpu_config_raw: str
    usd_per_hour: float
    available: bool
    recommended: bool = False


@dataclasses.dataclass(frozen=True)
class Deployment:
    deploy_id: str
    model_name: str
    status: str
    fail_reason: str = ""
    raw: Optional[Dict[str, Any]] = None


class DeepInfraDeployError(RuntimeError):
    pass


class DeepInfraDeployer:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_DEPLOY_BASE_URL,
        timeout_s: float = 60.0,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout_s),
        )

    async def _request_json(self, method: str, path: str, *, json_body: Any = None) -> Any:
        r = await self._client.request(method, path, json=json_body)
        if r.status_code >= 400:
            try:
                body = r.json()
            except Exception:
                body = r.text
            raise DeepInfraDeployError(f"{method} {path} failed status={r.status_code} body={body}")
        if r.status_code == 204:
            return None
        try:
            return r.json()
        except Exception:
            return r.text

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "DeepInfraDeployer":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def gpu_availability(self) -> List[GPUOption]:
        data = await self._request_json("GET", "/deploy/llm/gpu_availability")
        data = data or {}
        out: List[GPUOption] = []
        for item in data.get("gpus", []) or []:
            try:
                raw = str(item["gpu_config"])
                n, gpu = parse_gpu_config(raw)
                out.append(
                    GPUOption(
                        gpu=gpu,
                        num_gpus=n,
                        gpu_config_raw=raw,
                        usd_per_hour=float(item["usd_per_hour"]),
                        available=bool(item["available"]),
                        recommended=bool(item.get("recommended", False)),
                    )
                )
            except Exception:
                continue
        return out

    async def list_deployments(self) -> List[Dict[str, Any]]:
        data = await self._request_json("GET", "/deploy/list")
        return list(data or [])

    async def list_private_models(self) -> List[Dict[str, Any]]:
        data = await self._request_json("GET", "/models/private/list")
        return list(data or [])

    async def me(self) -> Dict[str, Any]:
        data = await self._request_json("GET", "/v1/me")
        return dict(data or {})

    async def infer_deploy_prefix(self) -> Optional[str]:
        """
        Best-effort inference of the DeepInfra namespace/prefix for custom deployments.
        """
        prefixes: List[str] = []

        try:
            for m in await self.list_private_models():
                name = str(m.get("model_name") or "")
                if "/" in name:
                    prefixes.append(name.split("/", 1)[0])
        except Exception:
            pass

        try:
            for d in await self.list_deployments():
                name = str(d.get("model_name") or "")
                if "/" in name:
                    prefixes.append(name.split("/", 1)[0])
        except Exception:
            pass

        if prefixes:
            # most common
            prefixes.sort()
            best = max(set(prefixes), key=prefixes.count)
            return best

        # fallback: uid
        try:
            me = await self.me()
            uid = str(me.get("uid") or "").strip()
            if uid and "/" not in uid:
                return uid
        except Exception:
            pass

        return None

    async def choose_gpu(
        self,
        *,
        preferred: Optional[Iterable[str]] = None,
        num_gpus: int = 1,
        allow_unavailable: bool = False,
    ) -> str:
        """
        Choose a GPU config string.

        Strategy:
        - If `preferred` provided: pick the first AVAILABLE GPU in that order
        - Else: pick a recommended AVAILABLE GPU (cheapest among recommended)
        - Else: pick the cheapest AVAILABLE GPU
        """
        gpus = await self.gpu_availability()
        candidates = [g for g in gpus if (g.available or allow_unavailable) and int(g.num_gpus) == int(num_gpus)]

        if preferred:
            pref = list(preferred)
            # strict preference order (default use-case: force B200 if available)
            for name in pref:
                for g in candidates:
                    if g.gpu == name and g.available:
                        return g.gpu

        # recommended first
        rec = [g for g in candidates if g.recommended and g.available]
        if rec:
            rec.sort(key=lambda g: g.usd_per_hour)
            return rec[0].gpu

        if not candidates:
            raise DeepInfraDeployError("No GPU options available from /deploy/llm/gpu_availability")
        candidates.sort(key=lambda g: g.usd_per_hour)
        return candidates[0].gpu

    async def deploy_llm_from_hf(
        self,
        *,
        deploy_model_name: str,
        hf_repo: str,
        gpu: str,
        num_gpus: int = 1,
        max_batch_size: int = 32,
        min_instances: int = 1,
        max_instances: int = 1,
        hf_revision: Optional[str] = None,
        hf_token: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
    ) -> Deployment:
        """
        Create an LLM deployment from HuggingFace weights.
        """
        payload: Dict[str, Any] = {
            "model_name": deploy_model_name,
            "gpu": parse_gpu_config(gpu)[1],
            "num_gpus": int(num_gpus),
            "max_batch_size": int(max_batch_size),
            "hf": {
                "repo": hf_repo,
                "revision": hf_revision,
                "token": hf_token,
            },
            "settings": {"min_instances": int(min_instances), "max_instances": int(max_instances)},
            "extra_args": extra_args,
        }
        # remove nulls to be safe
        if payload["hf"]["revision"] is None:
            payload["hf"].pop("revision", None)
        if payload["hf"]["token"] is None:
            payload["hf"].pop("token", None)
        if payload.get("extra_args") is None:
            payload.pop("extra_args", None)

        data = await self._request_json("POST", "/deploy/llm", json_body=payload)
        data = data or {}
        deploy_id = str(data.get("deploy_id") or data.get("deployId") or data.get("id") or "")
        if not deploy_id:
            raise DeepInfraDeployError(f"Unexpected deploy response (missing deploy_id): {data}")
        status = str(data.get("status") or "")
        fail_reason = str(data.get("fail_reason") or data.get("failReason") or "")
        model_name = str(data.get("model_name") or deploy_model_name)
        return Deployment(deploy_id=deploy_id, model_name=model_name, status=status, fail_reason=fail_reason, raw=data)

    async def get_deployment(self, deploy_id: str) -> Deployment:
        data = await self._request_json("GET", f"/deploy/{deploy_id}")
        data = data or {}
        return Deployment(
            deploy_id=str(data.get("deploy_id") or deploy_id),
            model_name=str(data.get("model_name") or ""),
            status=str(data.get("status") or ""),
            fail_reason=str(data.get("fail_reason") or ""),
            raw=data,
        )

    async def update_scale(self, deploy_id: str, *, min_instances: int, max_instances: int) -> None:
        payload = {"settings": {"min_instances": int(min_instances), "max_instances": int(max_instances)}}
        await self._request_json("PUT", f"/deploy/{deploy_id}", json_body=payload)

    async def delete(self, deploy_id: str) -> None:
        await self._request_json("DELETE", f"/deploy/{deploy_id}")

    async def wait_until_deployed(
        self,
        deploy_id: str,
        *,
        timeout_s: float = 60 * 30,
        poll_s: float = 6.0,
    ) -> Deployment:
        """
        Poll until status becomes ready, or fails/timeout.

        In practice DeepInfra may report:
        - "running" for active deployments
        - "deployed" for some legacy deployments
        """
        t0 = time.time()
        last: Optional[Deployment] = None
        last_status: Optional[str] = None
        while True:
            d = await self.get_deployment(deploy_id)
            last = d
            status = (d.status or "").strip().lower()
            if status != last_status:
                last_status = status
            if status in {"deployed", "running"}:
                return d
            if status in {"failed", "error"}:
                raise DeepInfraDeployError(f"Deployment failed: deploy_id={deploy_id} status={d.status} reason={d.fail_reason}")
            if time.time() - t0 > timeout_s:
                raise DeepInfraDeployError(f"Deployment timeout: deploy_id={deploy_id} last_status={d.status} reason={d.fail_reason}")
            await asyncio.sleep(poll_s)


def default_gpu_preference() -> Tuple[str, ...]:
    # Preferred (fastest) -> fallback.
    # User default: B200 single GPU.
    return ("B200-180GB", "H200-141GB", "H100-80GB", "A100-80GB", "L40S-48GB", "L4-24GB")


def parse_gpu_config(gpu_config: str) -> Tuple[int, str]:
    """
    DeepInfra /deploy/llm/gpu_availability may return configs like "1xA100-80GB".
    The deploy API expects:
      - gpu: one of DeployGPUs enum values like "A100-80GB"
      - num_gpus: integer
    """
    s = str(gpu_config or "").strip()
    if not s:
        return (1, "other")
    m = re.match(r"^(\\d+)x(.+)$", s)
    if m:
        try:
            n = int(m.group(1))
        except Exception:
            n = 1
        return (max(1, n), m.group(2))
    return (1, s)


