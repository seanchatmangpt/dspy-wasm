defmodule DspyWasm.Conformance do
  @moduledoc """
  Runs `consumer/conformance.json` against a host. The vectors are the same file
  the Python reference host runs; `expect` keys: `state`, `state_not`,
  `message_contains`, `equals`, `includes`.
  """

  @vectors Path.expand("../../priv/conformance.json", __DIR__)
  @external_resource @vectors

  def vectors, do: @vectors |> File.read!() |> Jason.decode!()

  @doc "Returns `%{state: \"ALIVE\" | \"FAILED\", passed: n, failed: [..]}`."
  def run(host, call_timeout \\ 120_000) do
    results =
      for v <- vectors() do
        name = "#{v["group"]}: #{v["name"]}"

        case DspyWasm.Host.call_json(host, v["export"], v["args"], call_timeout) do
          {:ok, report} ->
            case check(v["expect"], report) do
              :ok -> {:ok, name}
              {:error, why} -> {:error, name, why}
            end

          {:error, reason} ->
            {:error, name, "host error: #{inspect(reason)}"}
        end
      end

    failed = for {:error, name, why} <- results, do: {name, why}
    passed = Enum.count(results, &match?({:ok, _}, &1))
    %{state: if(failed == [], do: "ALIVE", else: "FAILED"), passed: passed, failed: failed}
  end

  def check(expect, report) do
    checks = [
      {"state", fn s -> report["state"] == s end},
      {"state_not", fn s -> report["state"] != s end},
      {"message_contains", fn m -> String.contains?(report["message"] || "", m) end},
      {"equals", fn e -> report == e end},
      {"includes", fn inc -> Enum.all?(inc, fn {k, v} -> report[k] == v end) end}
    ]

    Enum.find_value(checks, :ok, fn {key, fun} ->
      if Map.has_key?(expect, key) and not fun.(expect[key]),
        do:
          {:error,
           "expected #{key}=#{inspect(expect[key])}, got #{inspect(report) |> String.slice(0, 300)}"}
    end)
  end
end
