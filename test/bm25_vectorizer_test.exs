defmodule BM25VectorizerTest do
  use ExUnit.Case
  doctest Mighty.Preprocessing.BM25Vectorizer
  alias Mighty.Preprocessing.BM25Vectorizer

  setup do
    %{
      corpus: [
        "This is the first document",
        "This document is the second document",
        "And this is the third one",
        "Is this the first document"
      ]
    }
  end

  test "fits and transforms with default parameters", context do
    vectorizer =
      BM25Vectorizer.new()
      |> BM25Vectorizer.fit(context[:corpus])

    bm25_matrix = BM25Vectorizer.transform(vectorizer, context[:corpus])

    assert Nx.shape(bm25_matrix) == {4, 9}

    expected_vocab = %{
      "this" => 8,
      "is" => 3,
      "the" => 6,
      "first" => 2,
      "document" => 1,
      "second" => 5,
      "and" => 0,
      "third" => 7,
      "one" => 4
    }

    assert vectorizer.count_vectorizer.vocabulary == expected_vocab
    assert Nx.all(Nx.greater_equal(bm25_matrix, 0))

    doc_scores = Nx.slice_along_axis(bm25_matrix, 1, 1, axis: 1)
    [score1, score2, score3, score4] = Nx.to_flat_list(doc_scores)

    assert score2 == Enum.max([score1, score2, score3, score4])
    assert_in_delta score1, score4, 1.0e-6
    assert score3 == Enum.min([score1, score2, score3, score4])

    and_scores = Nx.slice_along_axis(bm25_matrix, 0, 1, axis: 1)
    [and_score1, and_score2, and_score3, and_score4] = Nx.to_flat_list(and_scores)

    assert and_score1 < 1.0e-9
    assert and_score2 < 1.0e-9
    assert and_score3 > 1.0
    assert and_score4 < 1.0e-9
  end

  test "fits and transforms with custom parameters", context do
    vectorizer =
      BM25Vectorizer.new(
        term_saturation_factor: 1.5,
        length_normalization_factor: 0.5,
        ngram_range: {1, 2}
      )
      |> BM25Vectorizer.fit(context[:corpus])

    bm25_matrix = BM25Vectorizer.transform(vectorizer, context[:corpus])

    {n_docs, n_features} = Nx.shape(bm25_matrix)
    assert n_docs == 4
    assert n_features > 9

    vocab = vectorizer.count_vectorizer.vocabulary
    assert Map.has_key?(vocab, "this is")
    assert Map.has_key?(vocab, "first document")

    assert Nx.all(Nx.greater_equal(bm25_matrix, 0))
  end

  test "BM25 scoring can be influenced with different hyperparameters",
       context do
    default_vectorizer = BM25Vectorizer.new() |> BM25Vectorizer.fit(context[:corpus])
    default_matrix = BM25Vectorizer.transform(default_vectorizer, context[:corpus])

    high_k1_vectorizer =
      BM25Vectorizer.new(term_saturation_factor: 2.0) |> BM25Vectorizer.fit(context[:corpus])

    high_k1_matrix = BM25Vectorizer.transform(high_k1_vectorizer, context[:corpus])

    low_b_vectorizer =
      BM25Vectorizer.new(length_normalization_factor: 0.5) |> BM25Vectorizer.fit(context[:corpus])

    low_b_matrix = BM25Vectorizer.transform(low_b_vectorizer, context[:corpus])

    # Check that scores are different with different parameters
    assert default_matrix != high_k1_matrix
    assert default_matrix != low_b_matrix

    # Focus on the term "document" (which is index 1 in our vocabulary)
    default_doc_scores = Nx.slice_along_axis(default_matrix, 1, 1, axis: 1)
    high_k1_doc_scores = Nx.slice_along_axis(high_k1_matrix, 1, 1, axis: 1)
    low_b_doc_scores = Nx.slice_along_axis(low_b_matrix, 1, 1, axis: 1)

    [def_score1, def_score2, _, _] = Nx.to_flat_list(default_doc_scores)
    [high_k1_score1, high_k1_score2, _, _] = Nx.to_flat_list(high_k1_doc_scores)
    [low_b_score1, low_b_score2, _, _] = Nx.to_flat_list(low_b_doc_scores)

    # Higher k1 (`term_saturation_factor`) should increase the impact of term frequency
    assert high_k1_score2 - high_k1_score1 > def_score2 - def_score1

    # Lower b (`length_normalization_factor`) should reduce the impact of document length normalization
    assert low_b_score1 != def_score1
    assert low_b_score2 != def_score2

    # Check that relative ordering is maintained
    assert high_k1_score2 > high_k1_score1
    assert low_b_score2 > low_b_score1
  end
end
