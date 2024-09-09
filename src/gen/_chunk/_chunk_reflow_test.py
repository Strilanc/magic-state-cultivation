import gen


def test_from_auto_rewrite_xs():
    result = gen.ChunkReflow.from_auto_rewrite(
        inputs=[
            gen.PauliMap({'X': [2, 3]}),
            gen.PauliMap({'X': [3, 4]}),
            gen.PauliMap({'X': [4, 5, 6]}),
            gen.PauliMap({'X': [5, 7]}),
            gen.PauliMap({'X': [8, 6]}),
            gen.PauliMap({'X': [7, 6]}),
        ],
        out2in={
            gen.PauliMap({'X': [2, 3]}): [gen.PauliMap({'X': [2, 3]})],
            gen.PauliMap({'X': [2]}): "auto",
        }
    )
    assert result == gen.ChunkReflow(out2in={
        gen.PauliMap({'X': [2, 3]}): [gen.PauliMap({'X': [2, 3]})],
        gen.PauliMap({'X': [2]}): [
            gen.PauliMap({'X': [2, 3]}),
            gen.PauliMap({'X': [3, 4]}),
            gen.PauliMap({'X': [4, 5, 6]}),
            gen.PauliMap({'X': [5, 7]}),
            gen.PauliMap({'X': [7, 6]}),
        ],
    })


def test_from_auto_rewrite_xyz():
    result = gen.ChunkReflow.from_auto_rewrite(
        inputs=[
            gen.PauliMap({'X': [2, 3]}),
            gen.PauliMap({'Z': [2, 3]}),
        ],
        out2in={
            gen.PauliMap({'Y': [2, 3]}): "auto",
        }
    )
    assert result == gen.ChunkReflow(out2in={
        gen.PauliMap({'Y': [2, 3]}): [
            gen.PauliMap({'X': [2, 3]}),
            gen.PauliMap({'Z': [2, 3]}),
        ],
    })


def test_from_auto_rewrite_keyed():
    result = gen.ChunkReflow.from_auto_rewrite(
        inputs=[
            gen.PauliMap({'X': [2, 3]}),
            gen.PauliMap({'Z': [2, 3]}).keyed('test'),
        ],
        out2in={
            gen.PauliMap({'Y': [2, 3]}): "auto",
        }
    )
    assert result == gen.ChunkReflow(out2in={
        gen.PauliMap({'Y': [2, 3]}): [
            gen.PauliMap({'X': [2, 3]}),
            gen.PauliMap({'Z': [2, 3]}).keyed('test'),
        ],
    })
