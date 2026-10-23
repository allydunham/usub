import shlex

import pytest

from usub import IndexRange, Job, UsubError

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

def test_plan_single():
    plan = Job(["python", "x.py"]).plan()
    assert plan.command == "python x.py" and plan.indices is None and not plan.is_array

def test_plan_range_keeps_placeholders_for_the_backend():
    plan = Job("run --seed {i} --n {}", array="1-4").plan()
    assert plan.command == "run --seed {i} --n {}"
    assert plan.indices == IndexRange(1, 4)

def test_plan_range_leaves_other_braces_alone():
    assert Job("cat {/} ${x}", array=3).plan().command == "cat {/} ${x}"

def test_plan_cmds_file_is_an_array_over_its_lines(tmp_path):
    path = tmp_path / "my cmds.txt"
    path.write_text("echo a\n\n# note\necho b")  # blank and comment lines are tasks too; no final newline
    plan = Job(cmds="my cmds.txt", cwd=tmp_path).plan()
    assert plan.command == f'sed -n "{{i}}p" {shlex.quote(str(path))} | bash'
    assert plan.indices == IndexRange(1, 4)

def test_plan_cmds_path_is_relative_to_cwd_unless_absolute(tmp_path):
    (tmp_path / "c.txt").write_text("true\n")
    assert str(tmp_path / "c.txt") in Job(cmds="c.txt", cwd=tmp_path).plan().command
    assert str(tmp_path / "c.txt") in Job(cmds=tmp_path / "c.txt", cwd="/elsewhere").plan().command

@pytest.mark.parametrize("template, expected", [
    ("cat {}", 'cat "$f"'),
    ("cat {/}", 'cat "$(basename "$f")"'),
    ("cat {//}", 'cat "$(dirname "$f")"'),
    ("cat {.}", 'cat "${f%.*}"'),
    ("cat {/.}", 'cat "$(basename "${f%.*}")"'),
    ("echo {i}: {}", 'echo {i}: "$f"'),  # the index is left for the backend to fill in
    ("gzip", 'gzip "$f"'),
    ("awk '{print $1}' {}", "awk '{print $1}' \"$f\""),
])
def test_plan_file_list_placeholders(template, expected, tmp_path):
    (tmp_path / "files.txt").write_text("data/a.fq.gz\ndata/b.fq.gz\n")
    plan = Job(template, file_list="files.txt", cwd=tmp_path).plan()
    assert plan.command == f'f=$(sed -n "{{i}}p" {tmp_path}/files.txt); {expected}'
    assert plan.indices == IndexRange(1, 2)

def test_plan_file_list_path_is_quoted(tmp_path):
    (tmp_path / "my files.txt").write_text("a\n")
    command = Job("gzip", file_list=tmp_path / "my files.txt").plan().command
    assert shlex.quote(str(tmp_path / "my files.txt")) in command

def test_list_file_ignores_trailing_blank_lines_only(tmp_path):
    (tmp_path / "c.txt").write_text("a\n\nb\n\n  \n")
    assert Job(cmds="c.txt", cwd=tmp_path).plan().indices == IndexRange(1, 3)  # the middle blank line is a task

@pytest.mark.parametrize("content", ["", "\n\n", "  \n"])
@pytest.mark.parametrize("kind", ["cmds", "file_list"])
def test_plan_empty_list_file(kind, content, tmp_path):
    (tmp_path / "c.txt").write_text(content)
    with pytest.raises(UsubError, match="c.txt is empty"):
        Job("x" if kind == "file_list" else None, cwd=tmp_path, **{kind: "c.txt"}).plan()

@pytest.mark.parametrize("job, message", [
    (Job(), "No command"),
    (Job("x", array=2, cmds="c"), r"Only one of array, cmds and file_list .*got array, cmds"),
    (Job("x", cmds="c", file_list="f"), r"Only one of .*got cmds, file_list"),
    (Job("x", array=2, file_list="f"), r"Only one of .*got array, file_list"),
    (Job(file_list="f"), "No command"),
    (Job("x", file_list="/nonexistent/f.txt"), "Cannot read /nonexistent/f.txt"),
    (Job("x", cmds="c"), "cannot be combined"),
    (Job(cmds="/nonexistent/c.txt"), "Cannot read /nonexistent/c.txt"),
])
def test_plan_errors(job, message):
    with pytest.raises(UsubError, match=message):
        job.plan()

@pytest.mark.parametrize("job, name", [
    (Job("/usr/bin/python3 x.py"), "python3"),
    (Job("FOO=1 cmd"), "FOO_1"),
    (Job(cmds="runs/my cmds.txt"), "my_cmds"),
    (Job("echo", cmds="x.txt"), "echo"),
    (Job("x", name="my job/1"), "my_job_1"),
    (Job("'unbalanced"), "usub"),
])
def test_default_name(job, name):
    assert job.default_name() == name

def test_str_shows_only_set_fields():
    assert str(Job()) == ""
    assert str(Job("echo hi", mem="2G", cores=0, setup=["module load x"], extra=[])) == (
        "command = 'echo hi'\ncores = 0\nmem = '2G'\nsetup = ['module load x']"
    )
