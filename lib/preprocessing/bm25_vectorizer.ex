defmodule Mighty.Preprocessing.BM25Vectorizer do
  @moduledoc """
  Best Match 25 (BM25) Vectorizer for `Mighty`.
  """

  alias Mighty.Preprocessing.CountVectorizer
  alias Mighty.Preprocessing.Shared

  defstruct [:count_vectorizer, :idf, :avg_doc_length, :doc_lengths, k1: 1.2, b: 0.75]

  @doc """
  Creates a new `BM25Vectorizer` struct with the given options.

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
      vectorizer.k1,
      vectorizer.b
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
       vectorizer.k1,
       vectorizer.b
     )}
  end

  defp calculate_idf(df, n_docs) do
    n_docs_tensor = Nx.broadcast(Nx.tensor(n_docs, type: Nx.type(df)), Nx.shape(df))
    Nx.log1p(Nx.divide(Nx.subtract(n_docs_tensor, df), Nx.add(df, 1)))
  end

  defp calculate_bm25_score(tf, idf, doc_lengths, avg_doc_length, k1, b) do
    doc_lengths = Nx.new_axis(doc_lengths, 1)
    len_norm = Nx.divide(doc_lengths, avg_doc_length)

    numerator = Nx.multiply(Nx.multiply(idf, tf), k1 + 1)
    denominator = Nx.add(tf, Nx.multiply(k1, Nx.add(1, Nx.multiply(b, Nx.subtract(len_norm, 1)))))

    Nx.sum(Nx.divide(numerator, denominator), axes: [1])
  end
end
