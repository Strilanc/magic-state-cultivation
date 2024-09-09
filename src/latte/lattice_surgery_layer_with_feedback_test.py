from latte.lattice_surgery_layer_with_feedback import \
    LatticeSurgeryLayerWithFeedback, ReplaceAction, LetAction, Expression, \
    DiscardShotAction


def test_measurement_and_replace_annotation():
    layer = LatticeSurgeryLayerWithFeedback.from_content("""
        replace A with Y else Z if mz2456 ^ mz2367 ^ mz0145
        replace B with Y else Z if mz1257
        replace C with Y else Z if mz2456 ^ mz2367
        replace D with Y else Z if mz1257 ^ mz2367
        measures mxy0
        measures mxy4
        measures mxy2
        measures mxy6

              /       /
         A   Z   B   Z
        /   /   /   / 
         .   .   .   .
              /       /
         C   Z   D   Z            
        /   /   /   /
    """)
    assert tuple(a.name for a in layer.measure_actions) == ('mxy0', 'mxy4', 'mxy2', 'mxy6')
    assert layer.replace_actions == (
        ReplaceAction(
            pattern='A',
            false_result='Z',
            true_result='Y',
            expression=Expression(variables=('mz2456', 'mz2367', 'mz0145'), op='^'),
        ),
        ReplaceAction(
            pattern='B',
            false_result='Z',
            true_result='Y',
            expression=Expression(variables=('mz1257',), op=''),
        ),
        ReplaceAction(
            pattern='C',
            false_result='Z',
            true_result='Y',
            expression=Expression(variables=('mz2456', 'mz2367'), op='^'),
        ),
        ReplaceAction(
            pattern='D',
            false_result='Z',
            true_result='Y',
            expression=Expression(variables=('mz1257', 'mz2367'), op='^'),
        ),
    )


def test_let_annotation():
    layer = LatticeSurgeryLayerWithFeedback.from_content("""
        measures mx1357
        let pass1 = mx1357 ^ mxy0 ^ mxy2 ^ mxy4 ^ mxy6
        discard_shot_if pass1

                      /
         X---X-@-Z---X
         |/ /    |  / 
         X   .   Z   .
                 |    / 
         .   X---Z---X            
            /       / 
    """)
    assert tuple(a.name for a in layer.measure_actions) == ('mx1357',)
    assert layer.let_actions == (
        LetAction(
            name='pass1',
            expression=Expression(variables=('mx1357', 'mxy0', 'mxy2', 'mxy4', 'mxy6'), op='^')
        ),
    )
    assert layer.checks == (DiscardShotAction(condition=Expression(variables=('pass1',), op='')),)
