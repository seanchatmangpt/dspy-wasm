defmodule DspyWasm.SignatureTest do
  use ExUnit.Case, async: true

  alias DspyWasm.{Requirement, Signature}

  @answer %{
    description: "Answer the question using the grounding passage.",
    inputs: [
      %{name: :question, type: :string, doc: "The question to answer.", required: true},
      %{name: :passage, type: :string, doc: nil, required: true}
    ],
    outputs: [%{name: :answer, type: :string, doc: nil, required: false}]
  }

  test "an ash_dspy signature becomes a typed dspy signature string with instructions" do
    assert {:ok, spec} = Signature.to_spec(@answer)
    assert spec["signature"] == "question: str, passage: str -> answer: str"
    assert spec["instructions"] =~ "Answer the question using the grounding passage."
    assert spec["instructions"] =~ "question: The question to answer."
  end

  test "arrays and scalar Ash types map to Python types" do
    assert {:ok, "list[str]"} = Signature.python_type({:array, :string})
    assert {:ok, "list[list[int]]"} = Signature.python_type({:array, {:array, :integer}})
    assert {:ok, "float"} = Signature.python_type(:decimal)
    assert {:ok, "str"} = Signature.python_type(:uuid)
  end

  test "an unmappable type is an error, never a guess" do
    assert {:error, {:unsupported_type, MyApp.Custom}} = Signature.python_type(MyApp.Custom)
    bad = put_in(@answer, [:outputs], [%{name: :answer, type: MyApp.Custom, doc: nil}])
    assert {:error, {:unsupported_type, MyApp.Custom}} = Signature.to_spec(bad)
  end

  test "field names that are not Python identifiers are refused" do
    for name <- [:"has-dash", :class, :"9x", :Question] do
      bad = put_in(@answer, [:inputs], [%{name: name, type: :string}])
      assert {:error, {:invalid_field_name, _}} = Signature.to_spec(bad)
    end
  end

  test "a signature needs inputs and outputs" do
    assert {:error, :no_inputs} = Signature.to_spec(%{@answer | inputs: []})
    assert {:error, :no_outputs} = Signature.to_spec(%{@answer | outputs: []})
    assert {:error, :malformed_signature} = Signature.to_spec(%{})
  end

  test "requirements compare percent bounds with the evaluate report's percent score" do
    req = %{dimension: :accuracy, operator: :gte, bound: 90}
    assert Requirement.satisfied?(req, 90.0)
    refute Requirement.satisfied?(req, 89.99)
    assert Requirement.satisfied?(%{operator: :lt, bound: 50}, 33.33)
    refute Requirement.satisfied?(%{operator: :eq, bound: 50}, 50.01)
    refute Requirement.satisfied?(%{operator: :gte, bound: 0}, nil)
  end

  test "a score comes only from an ALIVE evaluate report" do
    assert {:ok, 33.33} = Requirement.score(%{"state" => "ALIVE", "score" => 33.33})
    assert {:error, {:no_score, "FAILED"}} = Requirement.score(%{"state" => "FAILED"})
  end
end
