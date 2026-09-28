# Boots the real component (about a minute): set DSPY_WASM_PATH to a built dspy.wasm.
#   DSPY_WASM_PATH=../../dist/dspy.wasm mix test

defmodule DspyWasm.ConformanceTest do
  use ExUnit.Case, async: false

  @answer "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"
  @path System.get_env("DSPY_WASM_PATH")

  setup_all do
    if @path == nil, do: flunk("set DSPY_WASM_PATH to a built dspy.wasm (skips are failures here)")

    lm = fn _request -> {:ok, Jason.encode!(%{"text" => @answer})} end
    {:ok, host} = DspyWasm.Host.start_link(path: @path, lm: lm)
    assert :ok = DspyWasm.Host.await_ready(host)
    %{host: host}
  end

  test "the reference Elixir host passes every conformance vector", %{host: host} do
    report = DspyWasm.Conformance.run(host)
    assert report.failed == [], inspect(report.failed, pretty: true)
    assert report.passed == length(DspyWasm.Conformance.vectors())
  end

  test "a host whose component cannot boot reports :starting, not a crash" do
    {:ok, host} =
      DspyWasm.Host.start_link(path: "/nonexistent/dspy.wasm", lm: fn _ -> {:error, "unused"} end)

    assert {:error, _reason} = DspyWasm.Host.await_ready(host)
    assert {:error, :starting} = DspyWasm.Host.call_json(host, "component-version")
  end

  test "component-version is the contract's release version, a bare string not JSON", %{host: host} do
    assert {:ok, DspyWasm.Limits.version()} == DspyWasm.Host.call_json(host, "component-version")
  end

  test "a translated ash_dspy signature runs end to end", %{host: host} do
    {:ok, spec} =
      DspyWasm.Signature.to_spec(%{
        description: "Answer the question.",
        inputs: [%{name: :question, type: :string, doc: "The question.", required: true}],
        outputs: [%{name: :answer, type: :string, doc: nil, required: false}]
      })

    request = Map.merge(spec, %{"module" => "predict", "inputs" => %{"question" => "capital?"}})

    assert {:ok, %{"state" => "ALIVE", "outputs" => %{"answer" => "Paris"}}} =
             DspyWasm.run(host, request)
  end

  test "evaluate scores are percent and drive an ash_dspy requirement", %{host: host} do
    request = %{
      "program" => %{"signature" => "question -> answer"},
      "devset" => [
        %{"question" => "France?", "answer" => "Paris"},
        %{"question" => "Peru?", "answer" => "Lima"}
      ],
      "metric" => %{"name" => "exact_match", "field" => "answer"}
    }

    assert {:ok, report} = DspyWasm.evaluate(host, request)
    assert {:ok, 50.0} = DspyWasm.Requirement.score(report)
    assert DspyWasm.Requirement.satisfied?(%{operator: :gte, bound: 50}, 50.0)
    refute DspyWasm.Requirement.satisfied?(%{operator: :gte, bound: 90}, 50.0)
  end

  test "an LM-call budget aborts a compile that stays inside the request limits and the host recovers",
       %{host: _} do
    lm = fn _ -> {:ok, Jason.encode!(%{"text" => @answer})} end
    {:ok, host} = DspyWasm.Host.start_link(path: @path, lm: lm, budget: [max_lm_calls: 25])
    assert :ok = DspyWasm.Host.await_ready(host)

    request =
      Jason.encode!(%{
        "program" => %{"signature" => "question -> answer"},
        "optimizer" => "bootstrap-random-search",
        "trainset" => [%{"question" => "a", "answer" => "x"}, %{"question" => "b", "answer" => "y"}],
        "metric" => "exact_match",
        "config" => %{
          "num_candidate_programs" => 1000,
          "max_bootstrapped_demos" => 1,
          "max_labeled_demos" => 0
        }
      })

    started = System.monotonic_time(:millisecond)

    assert {:error, {:budget_exceeded, message}} =
             DspyWasm.Host.call_json(host, "compile", [request])

    assert message =~ "25 LM calls"
    assert System.monotonic_time(:millisecond) - started < 30_000

    assert :ok = DspyWasm.Host.await_ready(host)
    assert {:ok, %{"state" => "ALIVE"}} = DspyWasm.Host.call_json(host, "capabilities")
    assert %{boots: 2, boot_ms: boot_ms} = DspyWasm.Host.stats(host)
    IO.puts("second boot in the same VM: #{boot_ms} ms")
  end
end
