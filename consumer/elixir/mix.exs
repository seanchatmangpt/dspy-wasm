defmodule DspyWasm.MixProject do
  use Mix.Project

  def project do
    [
      app: :dspy_wasm,
      version: "26.9.28",
      elixir: "~> 1.14",
      deps: [{:wasmex, "~> 0.15"}, {:jason, "~> 1.4"}, {:telemetry, "~> 1.0"}],
      description: "Reference Elixir host for the dspy-wasm component (Wasmex)"
    ]
  end

  def application, do: [extra_applications: [:logger, :inets, :ssl, :public_key, :crypto]]
end
