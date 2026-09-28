defmodule DspyWasm.Signature do
  @moduledoc """
  Translate an Ash-style signature declaration into a dspy-wasm program spec.

  Takes plain maps (no Ash dependency), shaped like the `ash_dspy` entities:

      %{
        description: "Answer the question using the grounding passage.",
        inputs: [%{name: :question, type: :string, doc: "The question.", required: true}],
        outputs: [%{name: :answer, type: :string, doc: nil, required: false}]
      }

  and returns `{:ok, %{"signature" => "question: str, passage: str -> answer: str",
  "instructions" => "..."}}`, ready to merge into a `run` / `evaluate` / `compile`
  request as `module: "predict"` (or another module) plus the spec.

  dspy-wasm signatures are strings: a field's `doc` has no slot of its own, so
  the signature description and the field docs are folded into `instructions`.
  `required: false` inputs cannot be expressed (dspy has no optional field);
  they are still listed, and callers must supply them.
  """

  @scalar %{
    string: "str",
    ci_string: "str",
    atom: "str",
    uuid: "str",
    date: "str",
    time: "str",
    datetime: "str",
    utc_datetime: "str",
    utc_datetime_usec: "str",
    naive_datetime: "str",
    integer: "int",
    float: "float",
    decimal: "float",
    boolean: "bool",
    map: "dict",
    term: "str"
  }

  @spec to_spec(map()) :: {:ok, map()} | {:error, term()}
  def to_spec(%{inputs: inputs, outputs: outputs} = signature)
      when is_list(inputs) and is_list(outputs) do
    with :ok <- non_empty(inputs, :no_inputs),
         :ok <- non_empty(outputs, :no_outputs),
         {:ok, ins} <- fields(inputs),
         {:ok, outs} <- fields(outputs) do
      {:ok,
       %{"signature" => "#{Enum.join(ins, ", ")} -> #{Enum.join(outs, ", ")}"}
       |> put_instructions(instructions(signature, inputs ++ outputs))}
    end
  end

  def to_spec(_), do: {:error, :malformed_signature}

  @doc "The Python type dspy-wasm uses for an Ash type atom (or `{:array, type}`)."
  def python_type({:array, inner}) do
    with {:ok, t} <- python_type(inner), do: {:ok, "list[#{t}]"}
  end

  def python_type(type) when is_atom(type) do
    case Map.fetch(@scalar, type) do
      {:ok, t} -> {:ok, t}
      :error -> {:error, {:unsupported_type, type}}
    end
  end

  def python_type(other), do: {:error, {:unsupported_type, other}}

  defp non_empty([], reason), do: {:error, reason}
  defp non_empty(_, _), do: :ok

  defp fields(list) do
    Enum.reduce_while(list, {:ok, []}, fn field, {:ok, acc} ->
      with {:ok, name} <- field_name(field.name),
           {:ok, type} <- python_type(field.type) do
        {:cont, {:ok, acc ++ ["#{name}: #{type}"]}}
      else
        {:error, _} = error -> {:halt, error}
      end
    end)
  end

  defp field_name(name) when is_atom(name) and not is_nil(name),
    do: field_name(Atom.to_string(name))

  defp field_name(name) when is_binary(name) do
    if Regex.match?(~r/^[a-z_][a-z0-9_]*$/, name) and
         name not in ~w(class def if in is not or and None True False lambda),
       do: {:ok, name},
       else: {:error, {:invalid_field_name, name}}
  end

  defp field_name(other), do: {:error, {:invalid_field_name, other}}

  defp instructions(signature, fields) do
    docs =
      for %{doc: doc, name: name} <- fields, is_binary(doc) and doc != "", do: "#{name}: #{doc}"

    [Map.get(signature, :description), if(docs != [], do: "Fields:\n" <> Enum.join(docs, "\n"))]
    |> Enum.reject(&(&1 in [nil, ""]))
    |> Enum.join("\n\n")
  end

  defp put_instructions(spec, ""), do: spec
  defp put_instructions(spec, text), do: Map.put(spec, "instructions", text)
end
