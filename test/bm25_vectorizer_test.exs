defmodule BM25VectorizerTest do
  use ExUnit.Case
  doctest Mighty.Preprocessing.BM25Vectorizer
  alias Mighty.Preprocessing.BM25Vectorizer

  setup do
    %{
      corpus: [
        "This is the first document",
        "The brown fox and the purple pig danced in the moonlight",
        "This document is kinda cool",
        "And this is another document",
        "A fox and hound quickly walked into a bar",
        "Is this a document",
        "The hound crossed the road with a fox"
      ],
      query: "The quick brown fox and the hound crossed the road"
    }
  end

  test "creates a new BM25Vectorizer", _ do
    vectorizer =
      BM25Vectorizer.new(term_saturation_factor: 1.5, length_normalization_factor: 0.75)

    assert %BM25Vectorizer{term_saturation_factor: 1.5, length_normalization_factor: 0.75} =
             vectorizer
  end

  test "fits BM25Vectorizer to corpus", %{corpus: corpus} do
    vectorizer = BM25Vectorizer.new() |> BM25Vectorizer.fit(corpus)
    assert vectorizer.idf != nil
    assert vectorizer.avg_doc_length != nil
    assert vectorizer.doc_lengths != nil
  end

  test "transforms corpus using BM25Vectorizer", %{corpus: corpus} do
    vectorizer = BM25Vectorizer.new() |> BM25Vectorizer.fit(corpus)
    result = BM25Vectorizer.transform(vectorizer, corpus)
    assert Nx.shape(result) == {length(corpus)}
  end

  test "fit_transform on corpus", %{corpus: corpus} do
    {vectorizer, result} = BM25Vectorizer.new() |> BM25Vectorizer.fit_transform(corpus)
    assert vectorizer.idf != nil
    assert vectorizer.avg_doc_length != nil
    assert vectorizer.doc_lengths != nil
    assert Nx.shape(result) == {length(corpus)}
  end

  test "ranks documents based on query", %{corpus: corpus, query: query} do
    vectorizer =
      BM25Vectorizer.new(term_saturation_factor: 1.5, length_normalization_factor: 0.75)

    {fitted_vectorizer, _} = BM25Vectorizer.fit_transform(vectorizer, corpus)

    corpus_scores = BM25Vectorizer.transform(fitted_vectorizer, corpus)
    query_score = BM25Vectorizer.transform(fitted_vectorizer, [query])

    similarity = Nx.multiply(corpus_scores, Nx.broadcast(query_score, Nx.shape(corpus_scores)))
    ranked_indices = Nx.argsort(similarity, direction: :desc)
    highest_ranked_index = ranked_indices |> Nx.to_flat_list() |> List.first()
    highest_ranked = Enum.at(corpus, highest_ranked_index)

    assert highest_ranked == "The brown fox and the purple pig danced in the moonlight"
  end

  test "ranks documents based on query with stopwords", %{corpus: corpus, query: query} do
    # note the seemingly odd inclusion of "purple" for the sake of this test
    stopwords = ~w(a an and the is in of purple to for with)

    vectorizer =
      BM25Vectorizer.new(
        term_saturation_factor: 1.5,
        length_normalization_factor: 0.75,
        stop_words: stopwords
      )

    {fitted_vectorizer, _} = BM25Vectorizer.fit_transform(vectorizer, corpus)

    corpus_scores = BM25Vectorizer.transform(fitted_vectorizer, corpus)
    query_score = BM25Vectorizer.transform(fitted_vectorizer, [query])

    similarity = Nx.multiply(corpus_scores, Nx.broadcast(query_score, Nx.shape(corpus_scores)))
    ranked_indices = Nx.argsort(similarity, direction: :desc)
    ranked_scores = Nx.take(similarity, ranked_indices)

    ranked_results =
      ranked_indices
      |> Nx.to_flat_list()
      |> Enum.zip(Nx.to_flat_list(ranked_scores))
      |> Enum.with_index(1)
      |> Enum.map(fn {{index, score}, rank} ->
        {Enum.at(corpus, index), score, rank}
      end)

    assert {"A fox and hound quickly walked into a bar", highest_ranked_score, 1} =
             Enum.at(ranked_results, 0)

    assert highest_ranked_score >= 26.0
    assert {"Is this a document", _score, 7} = List.last(ranked_results)
  end
end
