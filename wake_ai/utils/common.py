"""Common utilities used across Wake AI."""

import enum
import sys
from pathlib import Path
from typing import Any

import jinja2
import jinja2.meta


def render_template(
    template: str | Path,
    context: dict[str, Any],
    *,
    name: str | None = None,
) -> str:
    """Render a Jinja2 template (string or file) against ``context``.

    This is the single rendering engine shared by workflow step prompts
    (:meth:`WorkflowStep.format_prompt`) and any statically-templated system
    prompt / instructions. Because both paths go through here, their behaviour
    is identical by construction.

    Behaviour:

    - ``jinja2.StrictUndefined`` is used, so a variable that is *used* in the
      template but absent from ``context`` raises ``jinja2.UndefinedError`` at
      render time (e.g. ``{{ missing }}``). The standard escape hatches still
      apply — ``{% if x is defined %}`` and ``{{ x | default(...) }}`` do not
      raise.
    - A non-fatal warning is printed for every variable referenced in the
      template that is missing from ``context`` (catches typos early).
    - If ``template`` is a :class:`~pathlib.Path`, it is read from disk and a
      ``FileSystemLoader`` rooted at its parent is installed so ``{% include %}``
      / ``{% extends %}`` resolve relative to the file.

    Args:
        template: Template text, or a path to a template file.
        context: Variables made available to the template.
        name: Optional label for the warning message (e.g. ``"step 'analyze'"``).
    """
    env = jinja2.Environment(
        undefined=jinja2.StrictUndefined,
        loader=jinja2.FileSystemLoader(str(template.parent)) if isinstance(template, Path) else None,
    )
    template_text = template.read_text() if isinstance(template, Path) else template

    # Parse first to surface missing variables as an early warning (the render
    # below is what actually raises, via StrictUndefined).
    ast = env.parse(template_text)
    for key in jinja2.meta.find_undeclared_variables(ast):
        if key not in context:
            where = f" in {name}" if name else ""
            print(f"Context key '{key}' used{where} not provided")

    return env.from_string(template_text).render(**context)


# Python version compatibility for StrEnum
if sys.version_info < (3, 11):
    class StrEnum(str, enum.Enum):
        """String enumeration for Python < 3.11 compatibility."""
        pass
else:
    class StrEnum(enum.StrEnum):
        """String enumeration using native StrEnum for Python >= 3.11."""
        pass