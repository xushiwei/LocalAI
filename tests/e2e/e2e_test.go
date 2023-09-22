package e2e_test

import (
	"context"
	"errors"
	"io"
	"os/exec"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	openaigo "github.com/otiai10/openaigo"
	"github.com/sashabaranov/go-openai"
)

var _ = Describe("E2E test", func() {
	var client *openai.Client
	var client2 *openaigo.Client

	Context("API with ephemeral models", func() {
		BeforeEach(func() {
			defaultConfig := openai.DefaultConfig("")
			defaultConfig.BaseURL = localAIURL

			client2 = openaigo.NewClient("")
			client2.BaseURL = defaultConfig.BaseURL

			// Wait for API to be ready
			client = openai.NewClientWithConfig(defaultConfig)
			Eventually(func() error {
				_, err := client.ListModels(context.TODO())
				return err
			}, "2m").ShouldNot(HaveOccurred())
		})

		// Check that the GPU was used
		AfterEach(func() {
			// Execute docker logs $$(docker ps -q --filter ancestor=localai-tests) as a command and check the output
			cmd := exec.Command("/bin/bash", "-xce", "docker logs $$(docker ps -q --filter ancestor=localai-tests)")
			out, err := cmd.CombinedOutput()
			Expect(err).ToNot(HaveOccurred())
			Expect(string(out)).To(ContainSubstring("found 1 CUDA devices"))
			Expect(string(out)).To(ContainSubstring("using CUDA for GPU acceleration"))
		})

		Context("Generates text", func() {
			It("streams chat tokens", func() {
				models, err := client.ListModels(context.TODO())
				Expect(err).ToNot(HaveOccurred())
				Expect(models.Models).ToNot(BeEmpty())

				model := models.Models[0].ID
				stream, err := client.CreateChatCompletionStream(context.TODO(), openai.ChatCompletionRequest{
					Model:    model,
					Messages: []openai.ChatCompletionMessage{{Content: "Can you count up to five?", Role: "user"}}})
				Expect(err).ToNot(HaveOccurred())
				defer stream.Close()

				tokens := 0
				text := ""
				for {
					response, err := stream.Recv()
					if errors.Is(err, io.EOF) {
						break
					}

					Expect(err).ToNot(HaveOccurred())
					text += response.Choices[0].Delta.Content
					tokens++
				}
				Expect(text).ToNot(BeEmpty())
				Expect(text).To(Or(ContainSubstring("Sure"), ContainSubstring("five")))

				Expect(tokens).ToNot(Or(Equal(1), Equal(0)))
			})
		})
	})
})
