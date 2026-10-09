import pytest

from usub import IndexRange, Job, UsubError
from usub.job import expand


@pytest.mark.parametrize("value, expected", [
    ("1-100", IndexRange(1, 100)), ("0-9:3", IndexRange(0, 9, 3)), ("5", IndexRange(1, 5)),
    (5, IndexRange(1, 5)), (range(2, 11, 2), IndexRange(2, 10, 2)), (" 3 - 4 ", IndexRange(3, 4)),
])
def test_index_range_parse(value, expected):
    assert IndexRange.parse(value) == expected


@pytest.mark.parametrize("value", ["", "a-b", "5-1", "1-10:0", "1,2,3", range(0)])
def test_index_range_invalid(value):
    with pytest.raises(UsubError):
        IndexRange.parse(value)


def test_index_range_len_and_spec():
    r = IndexRange(1, 10, 3)
    assert len(r) == 4
    assert r.spec() == "1-10:3"
    assert IndexRange(1, 5).spec() == "1-5"


@pytest.mark.parametrize("template, expected", [
    ("cat {}", "cat data/s1.fq.gz"),
    ("cat {/}", "cat s1.fq.gz"),
    ("cat {//}", "cat data"),
    ("cat {.}", "cat data/s1.fq"),
    ("cat {/.}", "cat s1.fq"),
    ("echo {i}: {}", "echo 7: data/s1.fq.gz"),
    ("gzip", "gzip data/s1.fq.gz"),
    ("awk '{print $1}' {}", "awk '{print $1}' data/s1.fq.gz"),
])
def test_expand(template, expected):
    assert expand(template, "data/s1.fq.gz", 7) == expected


def test_expand_quotes_unsafe_values():
    assert expand("wc -l {}", "my file's.txt", 1) == "wc -l 'my file'\"'\"'s.txt'"
    assert expand("ls {//}", "plain", 1) == "ls ."


def test_plan_single():
    plan = Job(["python", "x.py"]).plan()
    assert plan.command == "python x.py" and plan.tasks is None and not plan.is_array


def test_plan_range_substitutes_index():
    plan = Job("run --seed {i} --n {}", array="1-4").plan()
    assert plan.command == "run --seed ${USUB_INDEX} --n ${USUB_INDEX}"
    assert plan.indices == IndexRange(1, 4)


def test_plan_range_rejects_path_placeholders():
    with pytest.raises(UsubError, match="only valid with files"):
        Job("cat {/}", array=3).plan()


def test_plan_cmds_skips_blank_and_comments():
    plan = Job(cmds=["echo a", "", "  # note", " echo b "]).plan()
    assert plan.tasks == ("echo a", "echo b")
    assert plan.indices == IndexRange(1, 2)


def test_plan_files():
    plan = Job("gzip {}", files=["a", "b c"]).plan()
    assert plan.tasks == ("gzip a", "gzip 'b c'")


@pytest.mark.parametrize("job, message", [
    (Job(), "no command"),
    (Job("x", array=2, files=["a"]), "only one of"),
    (Job("x", cmds=["a"]), "cannot be combined"),
    (Job("x", files=[]), "no tasks"),
    (Job(cmds=["# only comments"]), "no tasks"),
])
def test_plan_errors(job, message):
    with pytest.raises(UsubError, match=message):
        job.plan()


@pytest.mark.parametrize("job, name", [
    (Job("/usr/bin/python3 x.py"), "python3"),
    (Job("FOO=1 cmd"), "FOO_1"),
    (Job(cmds=["a"]), "usub"),
    (Job("x", name="my job/1"), "my_job_1"),
    (Job("'unbalanced"), "usub"),
])
def test_default_name(job, name):
    assert job.default_name() == name
