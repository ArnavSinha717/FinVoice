"""Scoring primitives shared by the evaluation tasks."""

from dataclasses import dataclass


@dataclass
class Score:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def support(self) -> int:
        return self.tp + self.fn

    def row(self, label: str) -> str:
        return (f"  {label:<16} P {self.precision:5.3f}   R {self.recall:5.3f}   "
                f"F1 {self.f1:5.3f}   (tp={self.tp} fp={self.fp} fn={self.fn})")


def table(title: str, scores: dict[str, Score]) -> str:
    lines = [f"\n{title}", "─" * len(title)]
    total = Score()
    for label, s in sorted(scores.items()):
        lines.append(s.row(label))
        total.tp += s.tp; total.fp += s.fp; total.fn += s.fn
    lines.append("  " + "-" * 62)
    lines.append(total.row("MICRO-AVG"))
    return "\n".join(lines)
