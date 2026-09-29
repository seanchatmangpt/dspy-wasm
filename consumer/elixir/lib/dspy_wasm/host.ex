defmodule DspyWasm.Host do
  @moduledoc """
  A supervised host for one dspy-wasm component instance (Wasmex).

  What Wasmex 0.15.1 does not give an embedder, and what this module supplies:

    * **Budgets.** LM calls, tool calls and reply bytes are counted in the import
      functions. Past a budget the import raises, which traps the guest and aborts
      the call (`{:error, {:budget_exceeded, message}}`); the aborted instance
      cannot be re-entered, so the host boots a fresh one.
    * **Async boot.** Loading the 317 MB component takes about a minute (Wasmex
      compiles it on every start; the Python host reuses a compilation cache).
      `start_link/1` returns at once; `await_ready/2` blocks; calls made while
      booting return `{:error, :starting}`.
    * **Recycling.** After a budget abort, a call timeout, or `max_calls` calls,
      the instance is replaced.

  It does *not* supply an epoch deadline: Wasmex exposes none, and killing the
  component process does **not** stop a running guest (measured: the guest's OS
  threads kept running after `Process.exit(pid, :kill)`). Pure-Python loops in
  interpreted code are bounded inside the component (`max_interpreter_events`);
  a single long C call is not. Run the host under an OS-level CPU/time limit.

  Options: `:path` (component file, required), `:lm` (`(request_json -> {:ok, json}
  | {:error, message})`, required), `:tools` (map of name => fun, default
  `DspyWasm.Tools.builtin/0`), `:budget` (keyword: `:max_lm_calls`,
  `:max_tool_calls`, `:max_reply_bytes`, default the contract's), `:call_timeout`
  (ms, default 600_000), `:max_calls` (default `nil`), plus GenServer options.
  """

  use GenServer

  alias DspyWasm.{Limits, Tools}

  @lm_iface "chatman:dspy/lm@0.1.0"
  @tools_iface "chatman:dspy/tools@0.1.0"
  # Exports whose reply is a bare string rather than a JSON report.
  @plain_text ["component-version"]

  # ---------------------------------------------------------------- client

  def start_link(opts) do
    {gen_opts, opts} = Keyword.split(opts, [:name])
    GenServer.start_link(__MODULE__, opts, gen_opts)
  end

  @doc "Block until the component has booted. `:ok` or `{:error, reason}`."
  def await_ready(server, timeout \\ 300_000), do: GenServer.call(server, :await_ready, timeout)

  @doc """
  Call an export with JSON-string arguments and decode the JSON reply.

  `{:ok, map}` | `{:error, :starting}` | `{:error, {:budget_exceeded, msg}}` |
  `{:error, :timeout}` | `{:error, {:trap, msg}}`.
  """
  def call_json(server, export, args \\ [], timeout \\ :default) do
    GenServer.call(server, {:call, export, args, timeout}, :infinity)
  end

  @doc "`%{calls: since the last boot, boots: instances booted, boot_ms: last boot time}`."
  def stats(server), do: GenServer.call(server, :stats)

  # ---------------------------------------------------------------- server

  @impl true
  def init(opts) do
    state = %{
      path: Keyword.fetch!(opts, :path),
      lm: Keyword.fetch!(opts, :lm),
      tools: Keyword.get(opts, :tools, Tools.builtin()),
      budget: budget(Keyword.get(opts, :budget, [])),
      call_timeout: Keyword.get(opts, :call_timeout, 600_000),
      max_calls: Keyword.get(opts, :max_calls),
      component: nil,
      status: :booting,
      meter: nil,
      calls: 0,
      boots: 0,
      boot_ms: nil,
      boot_ref: nil,
      waiting: []
    }

    {:ok, state, {:continue, :boot}}
  end

  @impl true
  def handle_continue(:boot, state), do: {:noreply, start_boot(state)}

  @impl true
  def handle_call(:await_ready, from, %{status: :booting} = state),
    do: {:noreply, %{state | waiting: [from | state.waiting]}}

  def handle_call(:await_ready, _from, %{status: :ready} = state), do: {:reply, :ok, state}

  def handle_call(:await_ready, _from, %{status: {:failed, reason}} = state),
    do: {:reply, {:error, reason}, state}

  def handle_call(:stats, _from, state),
    do: {:reply, Map.take(state, [:calls, :boots, :boot_ms]), state}

  def handle_call({:call, _, _, _}, _from, %{status: status} = state) when status != :ready,
    do: {:reply, {:error, :starting}, state}

  def handle_call({:call, export, args, timeout}, _from, state) do
    reset(state.meter)
    timeout = if timeout == :default, do: state.call_timeout, else: timeout
    {result, state} = invoke(state, export, args, timeout)
    state = %{state | calls: state.calls + 1}

    if replace?(result, state) do
      {:reply, result, state |> retire() |> start_boot()}
    else
      {:reply, result, state}
    end
  end

  @impl true
  def handle_info({:booted, ref, result}, %{boot_ref: ref} = state) do
    case result do
      {:ok, pid, meter, boot_ms} ->
        Process.link(pid)

        state = %{
          state
          | component: pid,
            meter: meter,
            status: :ready,
            boots: state.boots + 1,
            boot_ms: boot_ms
        }

        {:noreply, reply_waiting(state, :ok)}

      {:error, reason} ->
        {:noreply, reply_waiting(%{state | status: {:failed, reason}}, {:error, reason})}
    end
  end

  def handle_info(_stale, state), do: {:noreply, state}

  defp invoke(state, export, args, timeout) do
    result =
      try do
        Wasmex.Components.call_function(state.component, export, args, timeout)
      catch
        :exit, {:timeout, _} -> :timeout
        :exit, reason -> {:exit, reason}
      end

    case {result, tripped(state)} do
      {_, message} when is_binary(message) -> {{:error, {:budget_exceeded, message}}, state}
      {{:ok, text}, nil} when export in @plain_text -> {{:ok, text}, state}
      {{:ok, json}, nil} -> {decode(json), state}
      {{:error, message}, nil} -> {{:error, {:trap, message}}, state}
      {:timeout, nil} -> {{:error, :timeout}, state}
      {{:exit, reason}, nil} -> {{:error, {:trap, inspect(reason)}}, state}
    end
  end

  defp decode(json) do
    case Jason.decode(json) do
      {:ok, map} -> {:ok, map}
      {:error, error} -> {:error, {:trap, "component returned invalid JSON: #{inspect(error)}"}}
    end
  end

  defp replace?({:error, _}, _state), do: true
  defp replace?({:ok, _}, %{max_calls: max, calls: calls}), do: max != nil and calls >= max

  defp retire(state) do
    if is_pid(state.component) and Process.alive?(state.component) do
      # Unlink first: the component is linked to this host, and killing it must not kill us.
      Process.unlink(state.component)
      Process.exit(state.component, :kill)
    end

    %{state | component: nil, calls: 0, status: :booting}
  end

  # Boots on a separate process so the host stays responsive (calls answer
  # {:error, :starting}). That process owns the link to the component and stays
  # alive for as long as the component does; the host links to the component too.
  defp start_boot(state) do
    host = self()
    ref = make_ref()

    spawn(fn ->
      case safe_boot(state) do
        {:ok, pid, meter, boot_ms} ->
          send(host, {:booted, ref, {:ok, pid, meter, boot_ms}})
          owner_loop(pid)

        {:error, reason} ->
          send(host, {:booted, ref, {:error, reason}})
      end
    end)

    %{state | boot_ref: ref, status: :booting}
  end

  defp owner_loop(pid) do
    mon = Process.monitor(pid)

    receive do
      {:DOWN, ^mon, _, _, _} -> :ok
    end
  end

  defp safe_boot(state) do
    do_boot(state)
  rescue
    error -> {:error, Exception.message(error)}
  catch
    :exit, reason -> {:error, {:exit, reason}}
  end

  defp do_boot(state) do
    meter = %{counters: :counters.new(4, []), trip: :atomics.new(1, [])}

    imports = %{
      @lm_iface => %{"complete" => {:fn, fn request -> lm(state, meter, request) end}},
      @tools_iface => %{"call" => {:fn, fn name, args -> tool(state, meter, name, args) end}}
    }

    started = System.monotonic_time(:millisecond)

    case Wasmex.Components.start_link(%{
           path: state.path,
           imports: imports,
           wasi: %Wasmex.Wasi.WasiP2Options{}
         }) do
      {:ok, pid} -> {:ok, pid, meter, System.monotonic_time(:millisecond) - started}
      {:error, reason} -> {:error, reason}
    end
  end

  defp reply_waiting(state, reply) do
    Enum.each(state.waiting, &GenServer.reply(&1, reply))
    %{state | waiting: []}
  end

  # ------------------------------------------------------------ the meter

  # counters: 1 = LM calls, 2 = tool calls, 3 = reply bytes; trip: the slot that overran (0 = none)
  @what %{1 => "LM calls", 2 => "tool calls", 3 => "reply bytes"}

  defp reset(%{counters: c, trip: trip}) do
    for i <- 1..3, do: :counters.put(c, i, 0)
    :atomics.put(trip, 1, 0)
  end

  defp tripped(state) do
    case :atomics.get(state.meter.trip, 1) do
      0 -> nil
      slot -> "guest call exceeded its budget of #{cap(state.budget, slot)} #{@what[slot]}"
    end
  end

  defp cap(budget, 1), do: budget.max_lm_calls
  defp cap(budget, 2), do: budget.max_tool_calls
  defp cap(budget, 3), do: budget.max_reply_bytes

  defp charge(meter, slot, amount, cap) do
    :counters.add(meter.counters, slot, amount)

    if cap != nil and :counters.get(meter.counters, slot) > cap do
      :atomics.put(meter.trip, 1, slot)
      raise "guest call exceeded its budget of #{cap} #{@what[slot]}"
    end
  end

  defp lm(state, meter, request) do
    charge(meter, 1, 1, state.budget.max_lm_calls)

    reply =
      case state.lm.(request) do
        {:ok, json} -> json
        {:error, message} -> Jason.encode!(%{"error" => to_string(message)})
      end

    charge(meter, 3, byte_size(reply), state.budget.max_reply_bytes)
    reply
  end

  defp tool(state, meter, name, args) do
    charge(meter, 2, 1, state.budget.max_tool_calls)
    reply = Tools.envelope(state.tools, name, args)
    charge(meter, 3, byte_size(reply), state.budget.max_reply_bytes)
    reply
  end

  defp budget(overrides) do
    %{
      max_lm_calls: Keyword.get(overrides, :max_lm_calls, Limits.fetch!("max_lm_calls")),
      max_tool_calls: Keyword.get(overrides, :max_tool_calls, Limits.fetch!("max_tool_calls")),
      max_reply_bytes: Keyword.get(overrides, :max_reply_bytes, Limits.fetch!("max_reply_bytes"))
    }
  end
end
