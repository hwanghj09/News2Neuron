"""Package frozen bridge file paths relative to the repo without changing hashes."""
import argparse
import json
from pathlib import Path

from news_flywire.provenance import sha256, write_json


def normalize(out):
    out = Path(out)
    protocol = out/'protocol.json'
    original = out/'protocol_original.json'
    if original.exists():
        raise ValueError('Original protocol already archived; do not rewrite packaging')
    data = json.loads(protocol.read_text())
    root = Path.cwd().resolve()
    normalized = {}
    for name, digest in data['files'].items():
        path = Path(name)
        target = str(path.relative_to(root)) if path.is_absolute() else name
        if sha256(target) != digest:
            raise ValueError('Frozen content changed: '+name)
        normalized[target] = digest
    before = sha256(protocol)
    original.write_bytes(protocol.read_bytes())
    data['files'] = normalized
    write_json(protocol, data)
    run = out/'run_manifest.json'
    if run.exists():
        (out/'run_manifest_original.json').write_bytes(run.read_bytes())
        m = json.loads(run.read_text())
        m['files'][str(protocol)] = sha256(protocol)
        write_json(run, m)
    write_json(out/'path_packaging.json', dict(original_protocol_sha256=before,
        packaged_protocol_sha256=sha256(protocol), original_archive=str(original),
        reason='Replace absolute repository paths with relative aliases; exact content hashes unchanged.',
        experiment_parameters_changed=False, selected_articles_changed=False,
        neural_algorithm_changed=False, outputs_changed=False))
    print('Packaged repository-relative paths; original frozen protocol preserved verbatim.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='results/article_bridge_v1')
    normalize(p.parse_args().out)
