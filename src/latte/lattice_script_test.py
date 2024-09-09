from latte.lattice_script import LatticeScript


def test_lattice_script_t_comparison():
    script = LatticeScript.from_str("""
          /   /
         T   T
    =====================================
        measures m0
              /
         Z-*-Z
        /   /
    =====================================
        replace A with Z else Y if m0
        measures m1
        discard_shot_if m1

        .    A
            / 
    """)
    hits = 0
    for _ in range(10):
        result, state = script.simulate()
        hits += result == 'correct'
        assert state.keys() == {
            'm0',
            'm1',
        }
    assert hits == 10


def test_lattice_script_t_comparison_bad():
    script = LatticeScript.from_str("""
          /   /
         T   T
    =====================================
        measures cmp
              /
         Z-*-Z
        /   /
    =====================================
        replace A with Y else Y if cmp
        measures parity
        discard_shot_if parity

        .    A
            / 
    """)

    for _ in range(100):
        result, state = script.simulate()
        if result != 'correct':
            break
    else:
        assert False, "All correct"


def test_lattice_script_actual_factory():
    script = LatticeScript.from_str("""
          /   /   /
         T   T   T   .
                      
         .   .   .   .
          /   /   /
         T   T   T   .
    =====================================
        measures mz2456

          /   /   /   /
         Z   Z   Z   T
        /   /   /|
         X-*-X---X   .
         |/  |/  |/   /
         Z   Z   Z   T            
        /   /   /
    =====================================
        measures mz1257

          /   /   /   /
         Z   Z   Z   Z
        /   /|  /|  /
         .   X---X-*-X
          /  |/   /  |/
         Z   Z   Z   Z            
        /   /   /   /
    =====================================
        measures mz0145
        measures mz2367

          /   /   /   /
         Z   Z   Z   Z
        /|  /|  /|  /|
         X-*-X   X-*-X
         |/  |/  |/  |/
         Z   Z   Z   Z            
        /   /   /   /
    =====================================
        let s0 = mz2456 ^ mz2367 ^ mz0145
        let s2 = mz1257
        let s4 = mz2456 ^ mz2367
        let s6 = mz1257 ^ mz2367
        replace A with Y else Z if s0
        replace B with Y else Z if s2
        replace C with Y else Z if s4
        replace D with Y else Z if s6
        measures mxy0
        measures mxy2
        measures mxy4
        measures mxy6

              /       /
         A   Z   B   Z
        /   /   /   / 
         .   .   .   .
              /       /
         C   Z   D   Z            
        /   /   /   /
    =====================================
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
    =====================================
        let f0 = mxy0 ^ mxy2 ^ mxy6
        let f1 = mxy0 ^ mxy4 ^ mxy6
        let f2 = mxy0 ^ mxy2 ^ mxy4
        feedback Z at 3 if f0
        feedback Z at 3+1j if f1
        feedback Z at 3+2j if f2
        feedback X at 3
        feedback X at 3+1j
        feedback X at 3+2j
        output_should_be CCZ

                     /
        .   .   .   X
                   / /
        X---X---X---X
       /             / 
        .   .   .   X            
                   / 
    """)
    result, state = script.simulate()
    assert state.keys() == {
        'mz2456',
        'mz1257',
        'mz0145',
        'mz2367',
        'mxy0',
        'mxy4',
        'mxy2',
        'mxy6',
        'mx1357',
        'pass1',
        's0',
        's2',
        's4',
        's6',
        'f0',
        'f1',
        'f2',
    }
    assert result == 'correct'
