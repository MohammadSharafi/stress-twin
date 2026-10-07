from stress_twin.llm import TokenFactory, load_dotenv


def test_dotenv_sets_missing_key_only(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nexport NEBIUS_API_KEY="from-file"\nNEBIUS_BASE_URL=http://127.0.0.1:1/v1/\nEMPTY=\n')
    for k in ("NEBIUS_API_KEY", "EMPTY"):  # set-then-delete so monkeypatch restores "absent" afterwards
        monkeypatch.setenv(k, "x")
        monkeypatch.delenv(k)
    monkeypatch.setenv("NEBIUS_BASE_URL", "http://already-set/v1/")
    load_dotenv([env])
    import os

    assert os.environ["NEBIUS_API_KEY"] == "from-file"
    assert os.environ["NEBIUS_BASE_URL"] == "http://already-set/v1/"  # existing values win
    assert "EMPTY" not in os.environ


def test_from_env_reads_dotenv_in_cwd(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("NEBIUS_API_KEY=abc123\n")
    monkeypatch.setenv("NEBIUS_API_KEY", "x")
    monkeypatch.delenv("NEBIUS_API_KEY")
    monkeypatch.chdir(tmp_path)
    tf = TokenFactory.from_env()
    assert tf.client.api_key == "abc123"


def test_no_key_leaks_between_tests():
    import os

    assert os.environ.get("NEBIUS_API_KEY") in (None, "test-key")
