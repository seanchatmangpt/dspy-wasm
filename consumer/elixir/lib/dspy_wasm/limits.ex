defmodule DspyWasm.Limits do
  @moduledoc """
  The numeric limits of the dspy-wasm contract, read from `consumer/contract.json`
  (generated from `limits.py`). Compile-time: a host built against one contract
  refuses to compile against a contract that lacks a limit.
  """

  @contract Path.expand("../../priv/contract.json", __DIR__)
  @external_resource @contract
  @decoded @contract |> File.read!() |> Jason.decode!()
  @limits Map.fetch!(@decoded, "limits")
  @version Map.fetch!(@decoded, "component_version")

  @doc "The component release version the contract describes (CalVer, `YY.M.D`)."
  def version, do: @version

  def all, do: @limits
  def fetch!(name), do: Map.fetch!(@limits, name)
end
