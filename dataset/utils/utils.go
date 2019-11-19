package utils

import (
	"fmt"
	"io"
	"io/ioutil"
	"math/rand"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/mholt/archiver"
	"github.com/sirupsen/logrus"
)

// ChunkStrings chunks based on cores
func ChunkStrings(array []string, numChunks int) [][]string {
	var divided [][]string
	chunkSize := (len(array) + numChunks - 1) / numChunks
	for i := 0; i < len(array); i += chunkSize {
		end := i + chunkSize
		if end > len(array) {
			end = len(array)
		}
		divided = append(divided, array[i:end])
	}
	return divided
}

// RandomString func generates random string of length n
func RandomString(n int) string {
	rand.Seed(time.Now().UnixNano())

	letters := []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
	b := make([]rune, n)
	for i := range b {
		b[i] = letters[rand.Intn(len(letters))]
	}
	return string(b)
}

// UnRar unrars file
func UnRar(src string, dest string) ([]string, error) {
	err := archiver.Unarchive(src, dest)
	if err != nil {
		return nil, err
	}

	return filepath.Glob(dest + "/*.dem")
}

// Contains checks if string array contains string
func Contains(arr []string, elem string) bool {
	for _, e := range arr {
		if e == elem {
			return true
		}
	}

	return false
}

// WriteFile writes a byte array to file
func WriteFile(body []byte, path string) error {
	if FileExists(path) {
		os.RemoveAll(path)
	}

	return ioutil.WriteFile(path, body, 0644)
}

// FileExists returns true if file exists
func FileExists(name string) bool {
	_, err := os.Stat(name)
	return !os.IsNotExist(err)
}

// WriteResponseToFile writes to file
func WriteResponseToFile(body io.Reader, fileName string) error {
	if FileExists(fileName) {
		logrus.Info("File already exists. Skipping...")
		return nil
	}

	out, err := os.Create(fileName)
	if err != nil {
		return err
	}

	defer out.Close()
	_, err = io.Copy(out, body)

	return err
}

// DownloadDemo downloads a demo and returns the reader
func DownloadDemo(url, filepath string) error {
	if FileExists(filepath) {
		logrus.Info("There is already a demo file. Skipping...")
		return nil
	}

	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()

	var netTransport = &http.Transport{
		Dial: (&net.Dialer{
			Timeout: 5 * time.Second,
		}).Dial,
		TLSHandshakeTimeout: 5 * time.Second,
	}

	var netClient = &http.Client{
		Timeout:   time.Second * 60 * 10,
		Transport: netTransport,
	}

	resp, err := netClient.Get(url)
	if err != nil {
		return err
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	_, err = io.Copy(out, resp.Body)
	if err != nil {
		return err
	}

	return nil
}
