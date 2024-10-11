defmodule Mighty.Preprocessing.BM25Vectorizer do
  @moduledoc """
  Best Match 25 (BM25) Vectorizer for `Mighty`.
  """

  alias Mighty.Preprocessing.CountVectorizer
  alias Mighty.Preprocessing.Shared

  defstruct [
    :count_vectorizer,
    :idf,
    :avg_doc_length,
    :doc_lengths,
    term_saturation_factor: 1.2,
    length_normalization_factor: 0.75
  ]

  @doc """
  Creates a new `BM25Vectorizer` struct with the given options.

  ## Options

    * `:term_saturation_factor` - Controls non-linear term frequency normalization (saturation).
      Higher values give more weight to term frequency. Default is `1.2`.
    * `:length_normalization_factor` - Controls document length normalization.
      Values closer to 1 give more value to document length. Default is `0.75`.

  _Also supports options found in `CountVectorizer`._

  Returns the new vectorizer.
  """
  def new(opts \\ []) do
    {general_opts, bm25_opts} =
      Keyword.split(opts, Shared.get_vectorizer_schema() |> Keyword.keys())

    count_vectorizer = CountVectorizer.new(general_opts)
    bm25_opts = Shared.validate_bm25!(bm25_opts)

    %__MODULE__{count_vectorizer: count_vectorizer}
    |> struct(bm25_opts)
  end

  @doc """
  Fits the vectorizer to the given corpus, computing its Term-Frequency matrix,
  Inverse Document Frequency, and average document length.

  Returns the fitted vectorizer.
  """
  def fit(%__MODULE__{count_vectorizer: count_vectorizer} = vectorizer, corpus) do
    {cv, tf} = CountVectorizer.fit_transform(count_vectorizer, corpus)
    df = Scholar.Preprocessing.binarize(tf) |> Nx.sum(axes: [0])

    {n_samples, _n_features} = Nx.shape(tf)
    idf = calculate_idf(df, n_samples)

    doc_lengths = Nx.sum(tf, axes: [1])
    avg_doc_length = Nx.mean(doc_lengths)

    struct(vectorizer,
      count_vectorizer: cv,
      idf: idf,
      avg_doc_length: avg_doc_length,
      doc_lengths: doc_lengths
    )
  end

  @doc """
  Transforms the given corpus into a BM25 matrix.

  If the vectorizer has not been fitted yet, it will raise an error.

  Returns the BM25 matrix of the corpus given a prior fitting.
  """
  def transform(%__MODULE__{count_vectorizer: count_vectorizer} = vectorizer, corpus) do
    tf = CountVectorizer.transform(count_vectorizer, corpus)
    doc_lengths = Nx.sum(tf, axes: [1])

    calculate_bm25_score(
      tf,
      vectorizer.idf,
      doc_lengths,
      vectorizer.avg_doc_length,
      vectorizer.term_saturation_factor,
      vectorizer.length_normalization_factor
    )
  end

  @doc """
  Fits the vectorizer to the given corpus and transforms it into a BM25 matrix.

  Returns a tuple with the fitted vectorizer and the transformed corpus.
  """
  def fit_transform(%__MODULE__{} = vectorizer, corpus) do
    vectorizer = fit(vectorizer, corpus)

    {vectorizer,
     calculate_bm25_score(
       CountVectorizer.transform(vectorizer.count_vectorizer, corpus),
       vectorizer.idf,
       vectorizer.doc_lengths,
       vectorizer.avg_doc_length,
       vectorizer.term_saturation_factor,
       vectorizer.length_normalization_factor
     )}
  end

  defp calculate_idf(df, n_docs) do
    n_docs_tensor = Nx.tensor(n_docs, type: Nx.type(df))

    df
    |> Nx.shape()
    |> then(&Nx.broadcast(n_docs_tensor, &1))
    |> Nx.subtract(df)
    |> Nx.divide(Nx.add(df, 1))
    |> Nx.log1p()
  end

  defp calculate_bm25_score(
         tf,
         idf,
         doc_lengths,
         avg_doc_length,
         term_saturation_factor,
         length_normalization_factor
       ) do
    doc_lengths = Nx.new_axis(doc_lengths, 1)
    len_norm = Nx.divide(doc_lengths, avg_doc_length)

    numerator = Nx.multiply(Nx.multiply(idf, tf), term_saturation_factor + 1)

    denominator =
      Nx.add(
        tf,
        Nx.multiply(
          term_saturation_factor,
          Nx.add(1, Nx.multiply(length_normalization_factor, Nx.subtract(len_norm, 1)))
        )
      )

    Nx.sum(Nx.divide(numerator, denominator), axes: [1])
  end
end
