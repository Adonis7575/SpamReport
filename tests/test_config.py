from spamfilter import config


def test_home_uses_env_override(spam_home):
    assert config.home() == spam_home
    assert spam_home.is_dir()


def test_subdirs_are_created(spam_home):
    assert config.corpora_dir() == spam_home / "corpora"
    assert config.datasets_dir().is_dir()
    assert config.models_dir().is_dir()


def test_settings_roundtrip():
    assert config.load_settings() == {}
    config.save_settings({"active_model": "C:/x/y.model"})
    assert config.load_settings() == {"active_model": "C:/x/y.model"}
