// Package medusa implements Medusa speculative decoding with online training.
// tree.go provides candidate tree construction and tree-based verification
// for multi-path speculative decoding.
package medusa

import (
	"math"
	"sort"
)

// CandidateNode represents a node in the candidate prediction tree.
// Each node corresponds to a token predicted by a Medusa head.
type CandidateNode struct {
	TokenID    int       // predicted token ID
	Confidence float32   // logit value (before softmax)
	HeadIdx    int       // which Medusa head predicted this (-1 for root)
	Depth      int       // depth in tree (0 for root children)
	Children   []*CandidateNode
	Parent     *CandidateNode
}

// CandidateTree is a tree of candidate token predictions built from
// Medusa head outputs. Each level corresponds to a Medusa head predicting
// the token at that future position.
type CandidateTree struct {
	Root     *CandidateNode // virtual root (no token)
	NumNodes int            // total nodes excluding root
}

// CandidatePath is a single root-to-leaf path through the candidate tree,
// representing one possible continuation sequence.
type CandidatePath struct {
	Tokens     []int   // token IDs along the path
	Confidence float32 // sum of logit confidences along the path
}

// BuildCandidateTree constructs a candidate tree from Medusa head logits.
// headLogits[i] contains the logit vector for head i (predicting position i+1).
// topK controls how many candidates per head (branching factor).
// maxNodes limits total tree size to prevent explosion with many heads.
//
// The tree structure:
//   - Level 0 (root children): top-k tokens from head 0
//   - Level 1: for each level-0 node, top-k tokens from head 1
//   - Level d: for each level-(d-1) node, top-k tokens from head d
func BuildCandidateTree(headLogits [][]float32, topK int, maxNodes int) *CandidateTree {
	if len(headLogits) == 0 {
		return nil
	}

	root := &CandidateNode{
		TokenID: -1,
		HeadIdx: -1,
		Depth:   -1,
	}

	tree := &CandidateTree{Root: root}

	// Pre-compute top-k for each head
	headTopK := make([][]int, len(headLogits))
	headTopV := make([][]float32, len(headLogits))
	for i, logits := range headLogits {
		if len(logits) == 0 {
			return nil
		}
		headTopK[i], headTopV[i] = topKWithValues(logits, topK)
	}

	// Build tree level by level using BFS
	currentLevel := []*CandidateNode{root}

	for headIdx := 0; headIdx < len(headLogits); headIdx++ {
		nextLevel := make([]*CandidateNode, 0, len(currentLevel)*topK)

		for _, parent := range currentLevel {
			for j := 0; j < len(headTopK[headIdx]); j++ {
				if tree.NumNodes >= maxNodes {
					goto done
				}

				child := &CandidateNode{
					TokenID:    headTopK[headIdx][j],
					Confidence: headTopV[headIdx][j],
					HeadIdx:    headIdx,
					Depth:      headIdx,
					Parent:     parent,
				}
				parent.Children = append(parent.Children, child)
				nextLevel = append(nextLevel, child)
				tree.NumNodes++
			}
		}

		currentLevel = nextLevel
	}

done:
	return tree
}

// Paths returns all root-to-leaf paths through the tree, sorted by
// aggregate confidence (descending). Each path represents one possible
// continuation sequence that can be verified against the target model.
func (t *CandidateTree) Paths() []CandidatePath {
	if t == nil || t.Root == nil {
		return nil
	}

	var paths []CandidatePath
	var dfs func(node *CandidateNode, tokens []int, conf float32)
	dfs = func(node *CandidateNode, tokens []int, conf float32) {
		if len(node.Children) == 0 {
			// Leaf node: emit path
			pathTokens := make([]int, len(tokens))
			copy(pathTokens, tokens)
			paths = append(paths, CandidatePath{
				Tokens:     pathTokens,
				Confidence: conf,
			})
			return
		}
		for _, child := range node.Children {
			dfs(child, append(tokens, child.TokenID), conf+child.Confidence)
		}
	}

	dfs(t.Root, nil, 0)

	// Sort by confidence descending
	sort.Slice(paths, func(i, j int) bool {
		return paths[i].Confidence > paths[j].Confidence
	})

	return paths
}

// Linearize returns the tree nodes in BFS order along with parent indices.
// tokens[i] is the token ID for linearized node i.
// parentIdx[i] is the index of node i's parent in the linearized array,
// or -1 for root children.
//
// This linearization enables construction of tree attention masks where
// each node attends only to its ancestors.
func (t *CandidateTree) Linearize() (tokens []int, parentIdx []int) {
	if t == nil || t.Root == nil {
		return nil, nil
	}

	// BFS traversal, skipping the virtual root
	type entry struct {
		node      *CandidateNode
		parentPos int // position of parent in output (-1 for root children)
	}

	queue := make([]entry, 0, t.NumNodes)
	for _, child := range t.Root.Children {
		queue = append(queue, entry{node: child, parentPos: -1})
	}

	tokens = make([]int, 0, t.NumNodes)
	parentIdx = make([]int, 0, t.NumNodes)

	head := 0
	for head < len(queue) {
		e := queue[head]
		myPos := head
		head++

		tokens = append(tokens, e.node.TokenID)
		parentIdx = append(parentIdx, e.parentPos)

		for _, child := range e.node.Children {
			queue = append(queue, entry{node: child, parentPos: myPos})
		}
	}

	return tokens, parentIdx
}

// BuildTreeAttentionMask creates a boolean attention mask for tree verification.
// mask[i*n+j] is true if node i should attend to node j.
// Each node attends to itself and all its ancestors (but not siblings or cousins).
//
// This mask enables verifying all tree candidates in a single forward pass
// when the attention mechanism supports custom masks.
func BuildTreeAttentionMask(n int, parentIdx []int) []bool {
	mask := make([]bool, n*n)

	for i := 0; i < n; i++ {
		// Each node attends to itself
		mask[i*n+i] = true

		// Walk up the parent chain to mark ancestors
		j := parentIdx[i]
		for j >= 0 {
			mask[i*n+j] = true
			j = parentIdx[j]
		}
	}

	return mask
}

// VerifyTreePath checks a single candidate path against target model logits.
// targetLogits[i] is the logit vector from the target model for position i.
// Returns the number of accepted tokens and the correction/bonus token.
//
// Acceptance: draft token at position i is accepted if argmax(targetLogits[i])
// equals the draft token. On first rejection, the target's preferred token
// becomes the final (correction) token.
func VerifyTreePath(path CandidatePath, targetLogits [][]float32, vocabSize int) (accepted int, finalToken int) {
	if len(path.Tokens) == 0 {
		return 0, 0
	}

	for i, draftToken := range path.Tokens {
		if i >= len(targetLogits) {
			break
		}

		targetToken := argmaxSlice(targetLogits[i])
		if draftToken == targetToken {
			accepted++
		} else {
			finalToken = targetToken
			return accepted, finalToken
		}
	}

	// All tokens accepted — sample bonus token from last position's logits
	if accepted == len(path.Tokens) && len(targetLogits) > len(path.Tokens) {
		finalToken = argmaxSlice(targetLogits[len(path.Tokens)])
	} else if accepted > 0 && accepted <= len(targetLogits) {
		finalToken = argmaxSlice(targetLogits[accepted-1])
	}

	return accepted, finalToken
}

// topKWithValues returns the indices and values of the k largest elements
// in the slice, sorted by value descending.
func topKWithValues(values []float32, k int) ([]int, []float32) {
	if k > len(values) {
		k = len(values)
	}

	type iv struct {
		idx int
		val float32
	}

	// For large vocabs with small k, use partial selection
	entries := make([]iv, len(values))
	for i, v := range values {
		entries[i] = iv{i, v}
	}

	// Partial sort: only need top-k
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].val > entries[j].val
	})

	indices := make([]int, k)
	vals := make([]float32, k)
	for i := 0; i < k; i++ {
		indices[i] = entries[i].idx
		vals[i] = entries[i].val
	}

	return indices, vals
}

// argmaxSlice returns the index of the maximum value in a float32 slice.
func argmaxSlice(values []float32) int {
	if len(values) == 0 {
		return 0
	}
	maxIdx := 0
	maxVal := float32(-math.MaxFloat32)
	for i, v := range values {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}
