defmodule DspyWasm.Requirement do
  @moduledoc """
  Check `ash_dspy`-style requirements (`dimension`, `operator`, integer `bound`)
  against an `evaluate` report.

  The aggregate `score` of an `evaluate` report is already a percentage
  (`0..100`, e.g. 33.33; per-example scores are `0..1`), and `ash_dspy` bounds
  are whole percent units, so `satisfied?/2` compares them directly.
  """

  @operators [:gte, :lte, :gt, :lt, :eq]

  def satisfied?(%{operator: op, bound: bound}, score) when op in @operators and is_number(score) do
    percent = score

    case op do
      :gte -> percent >= bound
      :lte -> percent <= bound
      :gt -> percent > bound
      :lt -> percent < bound
      :eq -> abs(percent - bound) < 1.0e-9
    end
  end

  def satisfied?(_, _), do: false

  @doc "Aggregate percent score from an `evaluate` report."
  def score(%{"state" => "ALIVE", "score" => score}) when is_number(score), do: {:ok, score}
  def score(report), do: {:error, {:no_score, report["state"]}}
end
