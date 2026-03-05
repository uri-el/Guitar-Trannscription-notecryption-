from __future__ import annotations
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional
import requests
KlangioFormat = Literal['xml', 'mxml', 'gp5', 'midi', 'midi_quant', 'pdf', 'json', 'audio']

@dataclass
class KlangioClient:
    api_key: str
    base_url: str = 'https://api.klang.io'
    timeout_s: int = 60

    def _headers(self) -> dict:
        return {'kl-api-key': self.api_key}

    def create_transcription_job(self, audio_path: str, *, model: str='universal', outputs: Iterable[str]=('mxml',), title: Optional[str]=None, composer: Optional[str]=None, webhook_url: Optional[str]=None) -> dict:
        url = f'{self.base_url}/transcription'
        params = {'model': model}
        if title:
            params['title'] = title
        if composer:
            params['composer'] = composer
        if webhook_url:
            params['webhook_url'] = webhook_url
        output_list = list(outputs)
        data = [('outputs', o) for o in output_list]
        with open(audio_path, 'rb') as f:
            files = {'file': f}
            resp = requests.post(url, headers=self._headers(), params=params, data=data, files=files, timeout=self.timeout_s)
        _raise_for_status(resp)
        return resp.json()

    def get_status(self, job_id: str) -> dict:
        url = f'{self.base_url}/job/{job_id}/status'
        resp = requests.get(url, headers=self._headers(), timeout=self.timeout_s)
        _raise_for_status(resp)
        return resp.json()

    def wait_for_completion(self, job_id: str, *, poll_s: float=2.0, max_wait_s: float=180.0) -> dict:
        t0 = time.time()
        while True:
            st = self.get_status(job_id)
            status = (st.get('status') or '').upper()
            if status in {'COMPLETED', 'FAILED'}:
                return st
            if time.time() - t0 > max_wait_s:
                raise TimeoutError(f'Job {job_id} not completed after {max_wait_s:.1f}s (last={st})')
            time.sleep(poll_s)

    def download_result(self, job_id: str, fmt: KlangioFormat, out_path: str, retries: int=12) -> str:
        url = f'{self.base_url}/job/{job_id}/{fmt}'
        headers = self._headers()
        last_err = None
        for attempt in range(retries):
            resp = requests.get(url, headers=headers, timeout=self.timeout_s)
            if resp.status_code == 200:
                outp = Path(out_path)
                outp.parent.mkdir(parents=True, exist_ok=True)
                outp.write_bytes(resp.content)
                return str(outp)
            if resp.status_code == 404 and attempt < retries - 1:
                wait = min(5 + attempt * 3, 30)
                print(f'[KLANGIO] {fmt} not ready (attempt {attempt+1}/{retries}), retrying in {wait}s...')
                time.sleep(wait)
                last_err = resp
                continue
            _raise_for_status(resp)
        if last_err is not None:
            try:
                payload = last_err.json()
            except Exception:
                payload = last_err.text
            raise RuntimeError(
                f'Klangio: format "{fmt}" not available after {retries} retries. '
                f'The file may not be supported for this transcription. '
                f'HTTP {last_err.status_code}: {payload}'
            )

def client_from_env() -> KlangioClient:
    key = os.environ.get('KLANGIO_API_KEY', '').strip()
    if not key:
        raise RuntimeError('Missing env var KLANGIO_API_KEY')
    return KlangioClient(api_key=key)

def _raise_for_status(resp: requests.Response) -> None:
    if resp.status_code == 429:
        raise RuntimeError('Rate limited (HTTP 429). On Basic plan you may be limited to 1 request/min.')
    if resp.status_code >= 400:
        try:
            payload = resp.json()
        except Exception:
            payload = resp.text
        raise RuntimeError(f'HTTP {resp.status_code}: {payload}')