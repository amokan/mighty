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
    :max_score,
    term_saturation_factor: 1.2,
    length_normalization_factor: 0.75,
    normalize: false
  ]

  @doc """
  Creates a new `BM25Vectorizer` struct with the given options.

  Returns the new vectorizer.

  ## Options

    * `:term_saturation_factor` (_often seen as 'k1'_) - Controls non-linear term frequency normalization (saturation).
      Higher values give more weight to term frequency. Default is `1.2`.
    * `:length_normalization_factor` (_often seen as 'b'_) - Controls document length normalization.
      Values closer to 1 give more value to document length. Default is `0.75`.
    * `:normalize` - Set to `true` to normalize the final BM25 scores to a range of 0..1. Default is `false`.

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
  def fit(
        %__MODULE__{count_vectorizer: count_vectorizer, normalize: normalize} = vectorizer,
        corpus
      ) do
    {cv, tf} = CountVectorizer.fit_transform(count_vectorizer, corpus)
    df = Scholar.Preprocessing.binarize(tf) |> Nx.sum(axes: [0])

    {n_samples, _n_features} = Nx.shape(tf)
    idf = calculate_idf(df, n_samples)

    doc_lengths = Nx.sum(tf, axes: [1])
    avg_doc_length = Nx.mean(doc_lengths)

    scores =
      calculate_bm25_score(
        tf,
        idf,
        doc_lengths,
        avg_doc_length,
        vectorizer.term_saturation_factor,
        vectorizer.length_normalization_factor
      )

    max_score = if normalize, do: Nx.reduce_max(scores), else: nil

    struct(vectorizer,
      count_vectorizer: cv,
      idf: idf,
      avg_doc_length: avg_doc_length,
      doc_lengths: doc_lengths,
      max_score: max_score
    )
  end

  @doc """
  Transforms the given corpus into a BM25 matrix.

  If the vectorizer has not been fitted yet, it will raise an error.

  Returns the BM25 matrix of the corpus given a prior fitting.
  """
  def transform(
        %__MODULE__{
          count_vectorizer: count_vectorizer,
          normalize: normalize,
          max_score: max_score
        } = vectorizer,
        corpus
      ) do
    tf = CountVectorizer.transform(count_vectorizer, corpus)
    doc_lengths = Nx.sum(tf, axes: [1])

    scores =
      calculate_bm25_score(
        tf,
        vectorizer.idf,
        doc_lengths,
        vectorizer.avg_doc_length,
        vectorizer.term_saturation_factor,
        vectorizer.length_normalization_factor
      )

    if normalize && max_score do
      Nx.divide(scores, max_score)
    else
      scores
    end
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
    |> Nx.shape()
    |> then(&Nx.broadcast(n_docs_tensor, &1))
    |> Nx.subtract(df)
    |> Nx.add(0.5)
    |> Nx.divide(Nx.add(df, 0.5))
    |> Nx.log()
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

    # numerator: (k1 + 1) * tf * idf
    numerator =
      tf
      |> Nx.multiply(idf)
      |> Nx.multiply(term_saturation_factor + 1)

    # denominator: k1 * (1 - b + b * len_norm) + tf
    denominator =
      len_norm
      |> Nx.subtract(1)
      |> Nx.multiply(length_normalization_factor)
      |> Nx.add(1)
      |> Nx.multiply(term_saturation_factor)
      |> Nx.add(tf)

    numerator
    |> Nx.divide(denominator)
    |> Nx.sum(axes: [1])
    |> Nx.max(1.0e-10)
  end
end
