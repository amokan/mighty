defmodule Mighty.Preprocessing.CountVectorizer do
  defstruct vocabulary: nil,
            fixed_vocabulary: false,
            ngram_range: {1, 1},
            max_features: nil,
            min_df: 1,
            max_df: 1.0,
            stop_words: [],
            binary: false,
            preprocessor: nil,
            tokenizer: nil

  defp make_ngrams(tokens, ngram_range) do
    {min_n, max_n} = ngram_range
    n_original_tokens = length(tokens)

    ngrams =
      for n <- min_n..min(max_n, n_original_tokens) do
        for i <- 0..(n_original_tokens - n) do
          Enum.slice(tokens, i, n) |> Enum.join(" ")
        end
      end

    ngrams |> Enum.flat_map(& &1)
  end

  defp do_process(vectorizer = %__MODULE__{}, doc) do
    {pre_mod, pre_func, pre_args} = vectorizer.preprocessor
    {token_mod, token_func, token_args} = vectorizer.tokenizer

    doc
    |> then(fn doc -> apply(pre_mod, pre_func, [doc | pre_args]) end)
    |> then(fn doc -> apply(token_mod, token_func, [doc | token_args]) end)
    |> make_ngrams(vectorizer.ngram_range)
    |> Enum.filter(fn token ->
      if not is_nil(vectorizer.stop_words), do: token not in vectorizer.stop_words, else: true
    end)
  end

  def build_vocab(vectorizer, corpus) do
    vocabulary =
      case vectorizer.vocabulary do
        nil ->
          corpus
          |> Enum.reduce([], fn doc, vocab ->
            vocab ++ do_process(vectorizer, doc)
          end)
          |> Enum.uniq()
          |> Enum.sort()
          |> Enum.with_index()
          |> Enum.into(%{})

        _ ->
          vectorizer.vocabulary
      end

    struct(vectorizer, vocabulary: vocabulary)
  end

  @doc """
  Creates a new CountVectorizer.
  Fits the vocabulary on the corpus if no vocabulary is provided.
  """
  def new(corpus, opts \\ []) do
    opts = Mighty.Preprocessing.Shared.validate_shared!(opts)
    fixed_vocab = if is_nil(opts[:vocabulary]), do: false, else: true
    vectorizer = %__MODULE__{fixed_vocabulary: fixed_vocab} |> struct(opts)
    build_vocab(vectorizer, corpus)
  end

  @doc """
  Transforms a corpus into a term frequency matrix and a document frequency matrix.

  The term frequency matrix is a matrix of shape (length(corpus), length(vocabulary) where each row
  is a document and each column is a feature. It represents the count of each feature in each document.

  The document frequency matrix is a vector of length (length(vocabulary)) where each element is the
  number of documents that contain the feature at the corresponding index in the vocabulary.
  """
  def transform(vectorizer = %__MODULE__{}, corpus) do
    idx_updates =
      corpus
      |> Enum.with_index()
      |> Enum.reduce([], fn {doc, doc_idx}, accum ->
        counts =
          doc
          |> then(&do_process(vectorizer, &1))
          |> Enum.reduce(
            Enum.map(vectorizer.vocabulary, fn {k, _} -> {k, 0} end) |> Enum.into(%{}),
            fn token, acc ->
              case Map.get(acc, token) do
                nil ->
                  Map.put(acc, token, 1)

                count ->
                  Map.put(acc, token, count + 1)
              end
            end
          )
          |> Enum.map(fn {k, v} -> [doc_idx, Map.get(vectorizer.vocabulary, k), v] end)

        counts ++ accum
      end)
      |> Enum.reverse()
      |> Nx.tensor()

    tf =
      Nx.broadcast(0, {length(corpus), Enum.count(vectorizer.vocabulary)})
      |> Nx.indexed_put(idx_updates[[.., 0..1]], idx_updates[[.., 2]])

    tf =
      case vectorizer.max_features do
        nil ->
          tf

        max_features ->
          tf
          |> Nx.sum(axes: [0])
          |> Nx.argsort(axis: 0, direction: :desc)
          |> then(&Nx.take(tf, &1, axis: 1))
          |> Nx.slice_along_axis(0, max_features, axis: 1)
      end

    tf =
      if vectorizer.binary do
        Nx.select(Nx.greater(tf, 0), 1, 0)
      else
        tf
      end

    df = Nx.select(Nx.greater(tf, 0), 1, 0)

    max_cond =
      case vectorizer.max_df do
        max_df when is_integer(max_df) ->
          Nx.less_equal(Nx.sum(df, axes: [0]), max_df)

        max_df when is_float(max_df) ->
          Nx.less_equal(Nx.mean(df, axes: [0]), max_df)

        _ ->
          raise ArgumentError, "max_df must be an integer or float in the range [0.0, 1.0]"
      end

    min_cond =
      case vectorizer.min_df do
        min_df when is_integer(min_df) ->
          Nx.greater_equal(Nx.sum(df, axes: [0]), min_df)

        min_df when is_float(min_df) ->
          Nx.greater_equal(Nx.mean(df, axes: [0]), min_df)

        _ ->
          raise ArgumentError, "min_df must be an integer or float in the range [0.0, 1.0]"
      end

    df = Nx.mean(df, axes: [0])

    true_values =
      Nx.logical_and(
        min_cond,
        max_cond
      )

    true_indices = Nx.argsort(true_values, axis: 0, direction: :desc)

    true_count = Nx.sum(true_values) |> Nx.to_number()

    tf =
      Nx.take(tf, true_indices, axis: 1)
      |> Nx.slice_along_axis(0, true_count, axis: 1)

    df =
      Nx.take(df, true_indices, axis: 0)
      |> Nx.slice_along_axis(0, true_count, axis: 0)

    {tf, df}
  end
end
