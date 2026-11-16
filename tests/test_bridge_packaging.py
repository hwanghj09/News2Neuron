import json
from pathlib import Path

import pytest

from news_flywire.provenance import sha256
from scripts.normalize_bridge_paths import normalize


def test_path_packaging_preserves_original_and_all_content_hashes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path/'source.txt'; source.write_text('frozen source')
    out = Path('run'); out.mkdir()
    original = json.dumps(dict(parameters={'rate':150}, files={str(source):sha256(source)}))
    (out/'protocol.json').write_text(original)
    normalize(out)
    assert (out/'protocol_original.json').read_text() == original
    changed = json.loads((out/'protocol.json').read_text())
    assert changed['parameters'] == {'rate':150}
    assert changed['files'] == {'source.txt':sha256(source)}
    with pytest.raises(ValueError, match='already archived'):
        normalize(out)


def test_changed_source_cannot_be_hidden_by_path_packaging(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path/'source.txt'; source.write_text('source')
    out = Path('run'); out.mkdir()
    original = json.dumps(dict(files={str(source):sha256(source)}))
    (out/'protocol.json').write_text(original)
    source.write_text('modified source')
    with pytest.raises(ValueError, match='Frozen content changed'):
        normalize(out)
    assert not (out/'protocol_original.json').exists()
    assert (out/'protocol.json').read_text() == original
