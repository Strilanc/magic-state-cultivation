import pathlib

import pytest

import gen
from latte.lattice_script import LatticeScript


def read_surgery_scripts() -> list[str]:
    factories_path = pathlib.Path(__file__).parent.parent.parent / 'testdata' / 'surgery_scripts'
    return list(str(e) for e in factories_path.iterdir() if str(e).endswith('.lat'))


@pytest.mark.parametrize('script_path', read_surgery_scripts())
def test_verify_factories(script_path: str):
    with open(script_path) as f:
        script = LatticeScript.from_str(f.read())
    p = pathlib.Path(script_path)

    gen.write_file(
        p.parent.parent.parent / 'out' / (p.name + '.html'),
        gen.viz_3d_gltf_model_html(script.to_3d_gltf_model(ignore_contradictions=True, spacing=3, wireframe=False)))
    gen.write_file(
        p.parent.parent.parent / 'out' / (p.name + '.zx.html'),
        gen.viz_3d_gltf_model_html(script.to_3d_gltf_model(ignore_contradictions=True, spacing=3, wireframe=True)))
    for _ in range(15):
        result, state = script.simulate()
        assert result == 'correct', (result, state)
