defmodule DspyWasm do
  @moduledoc """
  Thin facade over `DspyWasm.Host`: encode a request map, call the export, and
  return the decoded report. `{:ok, report}` is a transport success only; check
  `report["state"] == "ALIVE"` for the call's own outcome (a refused request is
  `%{"state" => "FAILED", "message" => ...}`).
  """

  alias DspyWasm.Host

  def capabilities(host), do: Host.call_json(host, "capabilities")

  def predict(host, signature, inputs),
    do: Host.call_json(host, "predict", [signature, Jason.encode!(inputs)])

  def run(host, request), do: Host.call_json(host, "run", [Jason.encode!(request)])
  def render(host, request), do: Host.call_json(host, "render", [Jason.encode!(request)])
  def evaluate(host, request), do: Host.call_json(host, "evaluate", [Jason.encode!(request)])
  def compile(host, request), do: Host.call_json(host, "compile", [Jason.encode!(request)])
end
