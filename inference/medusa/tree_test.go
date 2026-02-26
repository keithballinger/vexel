package medusa

import (
	"math"
	"testing"
)

func TestBuildCandidateTree(t *testing.T) {
	// 2 heads, each with vocab_size=8, top-k=2
	// Head 0 predicts position +1: token 3 (logit 5.0) and token 7 (logit 3.0)
	// Head 1 predicts position +2: token 1 (logit 4.0) and token 5 (logit 2.0)
	vocabSize := 8
	headLogits := make([][]float32, 2)
	for i := range headLogits {
		headLogits[i] = make([]float32, vocabSize)
	}

	// Head 0: token 3 best, token 7 second
	headLogits[0][3] = 5.0
	headLogits[0][7] = 3.0

	// Head 1: token 1 best, token 5 second
	headLogits[1][1] = 4.0
	headLogits[1][5] = 2.0

	tree := BuildCandidateTree(headLogits, 2, 32)

	if tree == nil {
		t.Fatal("BuildCandidateTree returned nil")
	}

	// Root should have 2 children (top-2 from head 0)
	if len(tree.Root.Children) != 2 {
		t.Fatalf("root has %d children, want 2", len(tree.Root.Children))
	}

	// First child should be token 3 (highest logit)
	if tree.Root.Children[0].TokenID != 3 {
		t.Errorf("root child 0 token = %d, want 3", tree.Root.Children[0].TokenID)
	}
	if tree.Root.Children[1].TokenID != 7 {
		t.Errorf("root child 1 token = %d, want 7", tree.Root.Children[1].TokenID)
	}

	// Each head-0 child should have 2 children from head 1
	for i, child := range tree.Root.Children {
		if len(child.Children) != 2 {
			t.Errorf("child %d has %d children, want 2", i, len(child.Children))
		}
		if child.Children[0].TokenID != 1 {
			t.Errorf("grandchild 0 of child %d = %d, want 1", i, child.Children[0].TokenID)
		}
		if child.Children[1].TokenID != 5 {
			t.Errorf("grandchild 1 of child %d = %d, want 5", i, child.Children[1].TokenID)
		}
	}
}

func TestBuildCandidateTreeMaxNodes(t *testing.T) {
	// 4 heads, top-k=3 would produce 3^4=81 nodes, but limit to 20
	headLogits := make([][]float32, 4)
	vocabSize := 16
	for i := range headLogits {
		headLogits[i] = make([]float32, vocabSize)
		for j := 0; j < vocabSize; j++ {
			headLogits[i][j] = float32(vocabSize - j) // descending
		}
	}

	tree := BuildCandidateTree(headLogits, 3, 20)

	if tree.NumNodes > 20 {
		t.Errorf("tree has %d nodes, want <= 20", tree.NumNodes)
	}
}

func TestCandidateTreePaths(t *testing.T) {
	// Build a simple 2-head tree with top-2
	vocabSize := 4
	headLogits := make([][]float32, 2)
	for i := range headLogits {
		headLogits[i] = make([]float32, vocabSize)
	}

	// Head 0: token 0 (logit=3.0), token 1 (logit=1.0)
	headLogits[0][0] = 3.0
	headLogits[0][1] = 1.0

	// Head 1: token 2 (logit=2.0), token 3 (logit=0.5)
	headLogits[1][2] = 2.0
	headLogits[1][3] = 0.5

	tree := BuildCandidateTree(headLogits, 2, 32)
	paths := tree.Paths()

	// Should have 4 paths: (0,2), (0,3), (1,2), (1,3)
	if len(paths) != 4 {
		t.Fatalf("expected 4 paths, got %d", len(paths))
	}

	// Paths should be sorted by confidence (descending)
	for i := 1; i < len(paths); i++ {
		if paths[i].Confidence > paths[i-1].Confidence {
			t.Errorf("paths not sorted: path[%d] confidence=%f > path[%d] confidence=%f",
				i, paths[i].Confidence, i-1, paths[i-1].Confidence)
		}
	}

	// Best path should be (0, 2) - highest combined confidence
	if paths[0].Tokens[0] != 0 || paths[0].Tokens[1] != 2 {
		t.Errorf("best path = %v, want [0, 2]", paths[0].Tokens)
	}
}

func TestCandidateTreeLinearize(t *testing.T) {
	// Build tree and linearize for verification
	vocabSize := 4
	headLogits := make([][]float32, 2)
	for i := range headLogits {
		headLogits[i] = make([]float32, vocabSize)
	}

	headLogits[0][0] = 3.0
	headLogits[0][1] = 1.0
	headLogits[1][2] = 2.0
	headLogits[1][3] = 0.5

	tree := BuildCandidateTree(headLogits, 2, 32)
	tokens, parentIdx := tree.Linearize()

	// Linearized tokens should include all unique tokens in the tree
	// BFS order: root children first, then grandchildren
	// Expected: [0, 1, 2, 3, 2, 3] (head0-top0, head0-top1, head1-children-of-0, head1-children-of-1)
	// But actually the root is implicit, so linearization is of children only.

	// All tokens from the tree should be present
	if len(tokens) == 0 {
		t.Fatal("linearize returned empty tokens")
	}

	// Parent indices: parentIdx[i] = index of parent in the linearized array, -1 for root children
	if len(parentIdx) != len(tokens) {
		t.Fatalf("parentIdx length %d != tokens length %d", len(parentIdx), len(tokens))
	}

	// Root children (head 0) should have parentIdx = -1
	for i := 0; i < len(tokens); i++ {
		if parentIdx[i] == -1 {
			// This is a root child (depth 1)
			continue
		}
		// Non-root nodes should point to a valid parent
		if parentIdx[i] < 0 || parentIdx[i] >= len(tokens) {
			t.Errorf("parentIdx[%d] = %d out of range", i, parentIdx[i])
		}
	}
}

func TestBuildAttentionMask(t *testing.T) {
	// Tree with 2 root children, each with 2 grandchildren = 6 nodes total
	vocabSize := 4
	headLogits := make([][]float32, 2)
	for i := range headLogits {
		headLogits[i] = make([]float32, vocabSize)
	}
	headLogits[0][0] = 3.0
	headLogits[0][1] = 1.0
	headLogits[1][2] = 2.0
	headLogits[1][3] = 0.5

	tree := BuildCandidateTree(headLogits, 2, 32)
	tokens, parentIdx := tree.Linearize()
	mask := BuildTreeAttentionMask(len(tokens), parentIdx)

	n := len(tokens)
	if len(mask) != n*n {
		t.Fatalf("mask size = %d, want %d", len(mask), n*n)
	}

	// Each node should attend to itself
	for i := 0; i < n; i++ {
		if !mask[i*n+i] {
			t.Errorf("node %d (token %d) should attend to itself", i, tokens[i])
		}
	}

	// Each node should attend to all its ancestors
	for i := 0; i < n; i++ {
		// Walk up the parent chain
		j := i
		for j >= 0 {
			if !mask[i*n+j] {
				t.Errorf("node %d should attend to ancestor %d", i, j)
			}
			j = parentIdx[j]
		}
	}

	// Siblings should NOT attend to each other
	// Find two root children (both have parentIdx == -1)
	rootChildren := []int{}
	for i := 0; i < n; i++ {
		if parentIdx[i] == -1 {
			rootChildren = append(rootChildren, i)
		}
	}
	if len(rootChildren) >= 2 {
		a, b := rootChildren[0], rootChildren[1]
		if mask[a*n+b] {
			t.Errorf("sibling %d should NOT attend to sibling %d", a, b)
		}
		if mask[b*n+a] {
			t.Errorf("sibling %d should NOT attend to sibling %d", b, a)
		}
	}
}

func TestTreeVerificationAcceptBestPath(t *testing.T) {
	// Simulate tree verification: target model confirms the best path
	vocabSize := 8
	headLogits := make([][]float32, 2)
	for i := range headLogits {
		headLogits[i] = make([]float32, vocabSize)
	}
	headLogits[0][3] = 5.0
	headLogits[0][7] = 3.0
	headLogits[1][1] = 4.0
	headLogits[1][5] = 2.0

	tree := BuildCandidateTree(headLogits, 2, 32)
	paths := tree.Paths()

	// Best path is [3, 1] (head0=3, head1=1)
	bestPath := paths[0]
	if bestPath.Tokens[0] != 3 || bestPath.Tokens[1] != 1 {
		t.Fatalf("best path = %v, want [3, 1]", bestPath.Tokens)
	}

	// Simulate target model verification:
	// targetLogits[0] predicts the first draft position, etc.
	targetLogits := make([][]float32, 2)
	for i := range targetLogits {
		targetLogits[i] = make([]float32, vocabSize)
	}
	// Target confirms token 3 at position 0
	targetLogits[0][3] = 10.0
	// Target confirms token 1 at position 1
	targetLogits[1][1] = 10.0

	accepted, finalToken := VerifyTreePath(bestPath, targetLogits, vocabSize)
	if accepted != 2 {
		t.Errorf("accepted = %d, want 2", accepted)
	}
	_ = finalToken // bonus token from last logits
}

func TestTreeVerificationRejectSecondToken(t *testing.T) {
	vocabSize := 8
	headLogits := make([][]float32, 2)
	for i := range headLogits {
		headLogits[i] = make([]float32, vocabSize)
	}
	headLogits[0][3] = 5.0
	headLogits[0][7] = 3.0
	headLogits[1][1] = 4.0
	headLogits[1][5] = 2.0

	tree := BuildCandidateTree(headLogits, 2, 32)
	paths := tree.Paths()
	bestPath := paths[0] // [3, 1]

	targetLogits := make([][]float32, 2)
	for i := range targetLogits {
		targetLogits[i] = make([]float32, vocabSize)
	}
	// Target confirms token 3 at position 0
	targetLogits[0][3] = 10.0
	// Target rejects token 1, prefers token 5 at position 1
	targetLogits[1][5] = 10.0

	accepted, finalToken := VerifyTreePath(bestPath, targetLogits, vocabSize)
	if accepted != 1 {
		t.Errorf("accepted = %d, want 1 (first token accepted, second rejected)", accepted)
	}
	if finalToken != 5 {
		t.Errorf("finalToken = %d, want 5 (correction from target)", finalToken)
	}
}

func TestTreeVerificationFallbackToAlternate(t *testing.T) {
	vocabSize := 8
	headLogits := make([][]float32, 2)
	for i := range headLogits {
		headLogits[i] = make([]float32, vocabSize)
	}
	headLogits[0][3] = 5.0
	headLogits[0][7] = 3.0
	headLogits[1][1] = 4.0
	headLogits[1][5] = 2.0

	tree := BuildCandidateTree(headLogits, 2, 32)
	paths := tree.Paths()

	// Best path [3, 1] is rejected at position 0 (target prefers 7)
	targetLogits := make([][]float32, 2)
	for i := range targetLogits {
		targetLogits[i] = make([]float32, vocabSize)
	}
	targetLogits[0][7] = 10.0 // Target wants 7, not 3
	targetLogits[1][1] = 10.0

	accepted, _ := VerifyTreePath(paths[0], targetLogits, vocabSize)
	if accepted != 0 {
		t.Errorf("best path accepted = %d, want 0 (rejected at position 0)", accepted)
	}

	// Try alternate path [7, 1] which should fully accept
	// Find path starting with 7
	var altPath CandidatePath
	for _, p := range paths {
		if p.Tokens[0] == 7 {
			altPath = p
			break
		}
	}

	if len(altPath.Tokens) == 0 {
		t.Fatal("could not find alternate path starting with token 7")
	}

	accepted2, _ := VerifyTreePath(altPath, targetLogits, vocabSize)
	if accepted2 != 2 {
		t.Errorf("alternate path accepted = %d, want 2", accepted2)
	}
}

func TestEmptyHeadLogits(t *testing.T) {
	tree := BuildCandidateTree(nil, 2, 32)
	if tree != nil {
		t.Error("expected nil tree for nil headLogits")
	}

	tree = BuildCandidateTree([][]float32{}, 2, 32)
	if tree != nil {
		t.Error("expected nil tree for empty headLogits")
	}
}

func TestSingleHeadTree(t *testing.T) {
	vocabSize := 4
	headLogits := [][]float32{make([]float32, vocabSize)}
	headLogits[0][2] = 5.0
	headLogits[0][0] = 3.0

	tree := BuildCandidateTree(headLogits, 2, 32)
	paths := tree.Paths()

	// With 1 head and top-2, we should have 2 paths of length 1
	if len(paths) != 2 {
		t.Fatalf("expected 2 paths, got %d", len(paths))
	}
	if len(paths[0].Tokens) != 1 {
		t.Errorf("path length = %d, want 1", len(paths[0].Tokens))
	}
	if paths[0].Tokens[0] != 2 {
		t.Errorf("best token = %d, want 2", paths[0].Tokens[0])
	}
}

func TestTopKExtraction(t *testing.T) {
	logits := []float32{1.0, 5.0, 3.0, 4.0, 2.0}
	indices, values := topKWithValues(logits, 3)

	if len(indices) != 3 || len(values) != 3 {
		t.Fatalf("got %d indices and %d values, want 3 each", len(indices), len(values))
	}

	// Should return indices sorted by value descending
	if indices[0] != 1 { // value 5.0
		t.Errorf("indices[0] = %d, want 1", indices[0])
	}
	if indices[1] != 3 { // value 4.0
		t.Errorf("indices[1] = %d, want 3", indices[1])
	}
	if indices[2] != 2 { // value 3.0
		t.Errorf("indices[2] = %d, want 2", indices[2])
	}

	if values[0] != 5.0 {
		t.Errorf("values[0] = %f, want 5.0", values[0])
	}
}

func TestTopKWithValuesSmallSlice(t *testing.T) {
	logits := []float32{3.0}
	indices, values := topKWithValues(logits, 5)

	if len(indices) != 1 {
		t.Errorf("got %d indices for 1-element slice, want 1", len(indices))
	}
	if values[0] != 3.0 {
		t.Errorf("values[0] = %f, want 3.0", values[0])
	}
}

func TestCandidatePathSorting(t *testing.T) {
	// Verify that paths are sorted by aggregate confidence
	vocabSize := 4
	headLogits := make([][]float32, 2)
	for i := range headLogits {
		headLogits[i] = make([]float32, vocabSize)
	}

	// Set up logits where alternative path has higher aggregate
	headLogits[0][0] = 3.0 // head 0 top-1
	headLogits[0][1] = 2.0 // head 0 top-2
	headLogits[1][2] = 1.0 // head 1 top-1
	headLogits[1][3] = 0.5 // head 1 top-2

	tree := BuildCandidateTree(headLogits, 2, 32)
	paths := tree.Paths()

	// Verify strictly descending confidence
	for i := 1; i < len(paths); i++ {
		if paths[i].Confidence > paths[i-1].Confidence+1e-6 {
			t.Errorf("paths[%d].Confidence (%f) > paths[%d].Confidence (%f)",
				i, paths[i].Confidence, i-1, paths[i-1].Confidence)
		}
	}
}

// TestVerifyTreePathEmpty tests edge case of empty path.
func TestVerifyTreePathEmpty(t *testing.T) {
	accepted, finalToken := VerifyTreePath(CandidatePath{}, nil, 8)
	if accepted != 0 {
		t.Errorf("accepted = %d, want 0 for empty path", accepted)
	}
	_ = finalToken
}

// TestTreeNumNodesAccurate tests that NumNodes accurately counts tree nodes.
func TestTreeNumNodesAccurate(t *testing.T) {
	vocabSize := 4
	headLogits := make([][]float32, 2)
	for i := range headLogits {
		headLogits[i] = make([]float32, vocabSize)
	}
	headLogits[0][0] = 3.0
	headLogits[0][1] = 1.0
	headLogits[1][2] = 2.0
	headLogits[1][3] = 0.5

	tree := BuildCandidateTree(headLogits, 2, 32)

	// 2 heads, top-2 = 2 + 2*2 = 6 nodes (excluding root)
	// Root has 2 children (head 0), each child has 2 children (head 1) = 2 + 4 = 6
	expected := 6
	if tree.NumNodes != expected {
		t.Errorf("NumNodes = %d, want %d", tree.NumNodes, expected)
	}
}

// BenchmarkBuildCandidateTree benchmarks tree construction.
func BenchmarkBuildCandidateTree(b *testing.B) {
	vocabSize := 32000
	numHeads := 4
	headLogits := make([][]float32, numHeads)
	for i := range headLogits {
		headLogits[i] = make([]float32, vocabSize)
		for j := 0; j < vocabSize; j++ {
			headLogits[i][j] = float32(j) * 0.001
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = BuildCandidateTree(headLogits, 3, 50)
	}
}

// Suppress unused import warning
var _ = math.MaxFloat32
