import pytest

from comfy_agent.config import PROJECT_ROOT, endpoint_url, load_settings, save_env, split_endpoint


def test_endpoint_helpers_and_env_override(tmp_path, monkeypatch):
    assert endpoint_url(" http://10.0.0.20/ ", "8188") == "http://10.0.0.20:8188"
    assert split_endpoint("http://10.0.0.15:11434") == ("10.0.0.15", 11434)
    for host, port in [("", 1), ("a b", 1), ("h", 0), ("h", "x"), ("h:1", 2)]:
        with pytest.raises(ValueError):
            endpoint_url(host, port)

    env = tmp_path / ".env"
    env.write_text("OPENROUTER_API_KEY=secret\n")
    monkeypatch.delenv("COMFYUI_URL", raising=False)
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    save_env({"COMFYUI_URL": "http://1.2.3.4:1", "OLLAMA_URL": "http://1.2.3.4:2"}, env)
    text = env.read_text()
    assert "OPENROUTER_API_KEY=secret" in text and "COMFYUI_URL=http://1.2.3.4:1" in text
    s = load_settings(PROJECT_ROOT / "config.yaml")
    assert s.comfyui_url == "http://1.2.3.4:1" and s.ollama_url == "http://1.2.3.4:2"
