defmodule DspyWasm.Tools do
  @moduledoc """
  Host tools with the contract's bounds: `calculator` (arithmetic only) and
  `embed` (deterministic hashing embedder). Each tool takes the decoded argument
  map and returns `{:ok, term}` or `{:error, "ErrorType: message"}`; `envelope/2`
  renders the JSON the component expects from `chatman:dspy/tools.call`.
  """

  alias DspyWasm.Limits

  def builtin do
    %{"calculator" => &calculator/1, "embed" => &embed/1, "echo" => &{:ok, &1}}
  end

  def envelope(tools, name, args_json) do
    with {:ok, fun} <-
           Map.fetch(tools, name) |> or_error("KeyError: unknown host tool #{inspect(name)}"),
         {:ok, args} when is_map(args) <-
           Jason.decode(args_json) |> or_error("TypeError: tool arguments must be a JSON object"),
         {:ok, result} <- safe(fun, args) do
      Jason.encode!(%{"result" => result})
    else
      {:error, message} -> Jason.encode!(%{"error" => message})
    end
  end

  defp or_error(:error, message), do: {:error, message}
  defp or_error({:error, _}, message), do: {:error, message}
  defp or_error(other, _), do: other

  defp safe(fun, args) do
    fun.(args)
  rescue
    error -> {:error, "#{inspect(error.__struct__)}: #{Exception.message(error)}"}
  end

  # ------------------------------------------------------------------ embed

  def embed(%{"texts" => texts} = args) when is_list(texts) do
    dims = Map.get(args, "dimensions", 64)
    max_dims = Limits.fetch!("max_embed_dimensions")
    max_texts = Limits.fetch!("max_embed_texts")
    max_values = Limits.fetch!("max_embed_values")

    cond do
      not Enum.all?(texts, &is_binary/1) ->
        {:error, "TypeError: texts must be a JSON array of strings"}

      not is_integer(dims) or dims < 1 or dims > max_dims ->
        {:error, "ValueError: dimensions must be an integer in [1, #{max_dims}]"}

      length(texts) > max_texts ->
        {:error, "ValueError: at most #{max_texts} texts per call"}

      length(texts) * dims > max_values ->
        {:error, "ValueError: texts x dimensions must not exceed #{max_values} values per call"}

      true ->
        {:ok, Enum.map(texts, &vector(&1, dims))}
    end
  end

  def embed(_), do: {:error, "TypeError: texts must be a JSON array of strings"}

  defp vector(text, dims) do
    counts =
      text
      |> String.downcase()
      |> String.split(~r/\W+/u, trim: true)
      |> Enum.frequencies_by(&:erlang.phash2(&1, dims))

    norm = :math.sqrt(counts |> Map.values() |> Enum.map(&(&1 * &1)) |> Enum.sum()) |> max(1.0)
    for i <- 0..(dims - 1), do: Map.get(counts, i, 0) / norm
  end

  # ------------------------------------------------------------- calculator

  def calculator(%{"expression" => expression}) when is_binary(expression) do
    with {:ok, tokens} <- tokenize(expression, []),
         {:ok, value, []} <- expr(tokens) do
      {:ok, value}
    else
      {:ok, _, rest} -> {:error, "SyntaxError: unexpected #{inspect(hd(rest))}"}
      {:error, _} = error -> error
    end
  end

  def calculator(_), do: {:error, "TypeError: expression must be a string"}

  defp tokenize("", acc), do: {:ok, Enum.reverse(acc)}
  defp tokenize(<<c, rest::binary>>, acc) when c in [?\s, ?\t], do: tokenize(rest, acc)
  defp tokenize(<<"**", rest::binary>>, acc), do: tokenize(rest, [:pow | acc])
  defp tokenize(<<"//", rest::binary>>, acc), do: tokenize(rest, [:floordiv | acc])

  defp tokenize(<<c, rest::binary>>, acc) when c in [?+, ?-, ?*, ?/, ?%, ?(, ?)] do
    op = %{?+ => :add, ?- => :sub, ?* => :mul, ?/ => :div, ?% => :mod, ?( => :lp, ?) => :rp}[c]
    tokenize(rest, [op | acc])
  end

  defp tokenize(<<c, _::binary>> = input, acc) when c in ?0..?9 or c == ?. do
    case Regex.run(~r/^(\d+\.\d*|\.\d+|\d+)/, input) do
      [number | _] ->
        value =
          if String.contains?(number, "."),
            do: String.to_float(normalize(number)),
            else: String.to_integer(number)

        tokenize(binary_part(input, byte_size(number), byte_size(input) - byte_size(number)), [
          {:num, value} | acc
        ])

      _ ->
        {:error, "SyntaxError: bad number"}
    end
  end

  defp tokenize(<<c::utf8, _::binary>>, _), do: {:error, "SyntaxError: unexpected #{<<c::utf8>>}"}

  defp normalize("." <> rest), do: "0." <> rest
  defp normalize(number), do: if(String.ends_with?(number, "."), do: number <> "0", else: number)

  # expr := term (('+'|'-') term)*
  defp expr(tokens) do
    with {:ok, left, rest} <- term(tokens), do: expr_tail(left, rest)
  end

  defp expr_tail(left, [op | rest]) when op in [:add, :sub] do
    with {:ok, right, rest} <- term(rest),
         {:ok, value} <- bounded(if(op == :add, do: left + right, else: left - right)),
         do: expr_tail(value, rest)
  end

  defp expr_tail(left, rest), do: {:ok, left, rest}

  # term := unary (('*'|'/'|'//'|'%') unary)*
  defp term(tokens) do
    with {:ok, left, rest} <- unary(tokens), do: term_tail(left, rest)
  end

  defp term_tail(left, [op | rest]) when op in [:mul, :div, :floordiv, :mod] do
    with {:ok, right, rest} <- unary(rest),
         {:ok, value} <- arith(op, left, right),
         do: term_tail(value, rest)
  end

  defp term_tail(left, rest), do: {:ok, left, rest}

  defp arith(:mul, a, b), do: bounded(a * b)

  defp arith(op, _, b) when op in [:div, :floordiv, :mod] and b == 0,
    do: {:error, "ZeroDivisionError: division by zero"}

  defp arith(:div, a, b), do: {:ok, a / b}

  defp arith(:floordiv, a, b) when is_integer(a) and is_integer(b),
    do: {:ok, Integer.floor_div(a, b)}

  defp arith(:floordiv, a, b), do: {:ok, Float.floor(a / b)}
  defp arith(:mod, a, b) when is_integer(a) and is_integer(b), do: {:ok, Integer.mod(a, b)}
  defp arith(:mod, a, b), do: {:ok, a - b * Float.floor(a / b)}

  defp unary([:sub | rest]) do
    with {:ok, v, rest} <- unary(rest), do: {:ok, -v, rest}
  end

  defp unary([:add | rest]), do: unary(rest)
  defp unary(tokens), do: power(tokens)

  # power := atom ['**' unary]   (right associative, binds tighter than unary on its left)
  defp power(tokens) do
    with {:ok, base, rest} <- atom(tokens) do
      case rest do
        [:pow | rest] ->
          with {:ok, exp, rest} <- unary(rest),
               {:ok, value} <- pow(base, exp),
               do: {:ok, value, rest}

        _ ->
          {:ok, base, rest}
      end
    end
  end

  defp atom([{:num, n} | rest]), do: {:ok, n, rest}

  defp atom([:lp | rest]) do
    case expr(rest) do
      {:ok, v, [:rp | rest]} -> {:ok, v, rest}
      {:ok, _, _} -> {:error, "SyntaxError: expected ')'"}
      error -> error
    end
  end

  defp atom(_), do: {:error, "SyntaxError: invalid expression"}

  # Refuse before computing: the size of an integer power is known from its operands.
  defp pow(base, exp) when is_integer(base) and is_integer(exp) and exp >= 0 do
    bits = Limits.fetch!("max_int_bits")

    if base != 0 and bit_length(base) * exp > bits do
      {:error, "ValueError: result exceeds #{bits} bits"}
    else
      {:ok, Integer.pow(base, exp)}
    end
  end

  defp pow(base, exp) when is_number(base) and is_number(exp) do
    {:ok, :math.pow(base, exp)}
  rescue
    _ -> {:error, "ValueError: result too large"}
  end

  defp bit_length(n), do: n |> abs() |> Integer.digits(2) |> length()

  defp bounded(value) when is_integer(value) do
    bits = Limits.fetch!("max_int_bits")

    if bit_length(value) > bits,
      do: {:error, "ValueError: result exceeds #{bits} bits"},
      else: {:ok, value}
  end

  defp bounded(value), do: {:ok, value}
end
