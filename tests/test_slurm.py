from usub.backends.slurm import slurm_time

def test_formatting():
    assert slurm_time(5415) == "01:30:15"
    assert slurm_time(216000) == "2-12:00:00"
