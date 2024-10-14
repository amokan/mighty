defmodule Mighty.Preprocessing.BM25Vectorizer do
  @moduledoc """
  Okapi BM25 (Best Match 25) Vectorizer for `Mighty`.
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

  @epsilon 1.0e-10

  @doc """
  Creates a new `BM25Vectorizer` struct with the given options.

  Returns the new vectorizer.

  ## Options

    * `:term_saturation_factor` (_often shown as 'k1'_) - Controls non-linear term frequency normalization (saturation).
      Higher values give more weight to term frequency. Default is `1.2`.
    * `:length_normalization_factor` (_often shown as 'b'_) - Controls document length normalization.
      Values closer to 1 give more value to document length. Default is `0.75`.

  _Also supports options found in `CountVectorizer`._
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
      avg_doc_length: avg_doc_length
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
    fitted_vectorizer = fit(vectorizer, corpus)
    {fitted_vectorizer, transform(fitted_vectorizer, corpus)}
  end

  defp calculate_idf(df, n_docs) do
    n_docs_tensor = Nx.tensor(n_docs, type: Nx.type(df))

    df
    |> Nx.add(@epsilon)
    |> Nx.divide(n_docs_tensor)
    |> Nx.log()
    |> Nx.subtract(1)
    |> Nx.multiply(-1)
  end

  defp calculate_bm25_score(
         tf,
         idf,
         doc_lengths,
         avg_doc_length,
         term_saturation_factor,
         length_normalization_factor
       ) do
    # doc length normalization
    len_norm =
      doc_lengths
      |> Nx.new_axis(1)
      |> Nx.divide(avg_doc_length)
      |> Nx.add(@epsilon)

    # numerator: (k1 + 1) * tf
    numerator = Nx.multiply(tf, term_saturation_factor + 1)

    # denominator: k1 * (1 - b + b * len_norm) + tf
    denominator =
      len_norm
      |> Nx.subtract(1)
      |> Nx.multiply(length_normalization_factor)
      |> Nx.add(1)
      |> Nx.multiply(term_saturation_factor)
      |> Nx.add(tf)
      |> Nx.add(@epsilon)

    numerator
    |> Nx.divide(denominator)
    |> Nx.multiply(idf)
    |> Nx.max(@epsilon)
  end
end
