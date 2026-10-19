// vision_helper.swift - tiny macOS helper around Apple's Vision framework.
//
// Usage:  vision_helper <image> [<image> ...]
// Output: JSON array, one object per image:
//   {"path": "...", "classifications": [{"id": "volleyball", "confidence": 0.83}, ...],
//    "text": ["..."]}
//
// Two requests are run per image:
//   * VNClassifyImageRequest  - Apple's built-in ~1300-class image classifier
//   * VNRecognizeTextRequest  - OCR (Japanese + English) for scoreboards,
//                               banners and venue signage.
//
// Nothing leaves the machine: both requests are on-device Core ML models.

import Foundation
import Vision
import ImageIO
import CoreGraphics

func emit(_ object: Any) {
    let data = try! JSONSerialization.data(withJSONObject: object, options: [])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
}

let args = Array(CommandLine.arguments.dropFirst())
if args.isEmpty {
    FileHandle.standardError.write("usage: vision_helper <image> [<image> ...]\n".data(using: .utf8)!)
    exit(2)
}

var results: [[String: Any]] = []

for path in args {
    var entry: [String: Any] = ["path": path]

    guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        entry["error"] = "cannot decode image"
        results.append(entry)
        continue
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    let classifyRequest = VNClassifyImageRequest()

    var recognizedText: [String] = []
    let textRequest = VNRecognizeTextRequest { request, _ in
        guard let observations = request.results as? [VNRecognizedTextObservation] else { return }
        for observation in observations {
            if let candidate = observation.topCandidates(1).first {
                recognizedText.append(candidate.string)
            }
        }
    }
    textRequest.recognitionLevel = .accurate
    textRequest.usesLanguageCorrection = false
    textRequest.recognitionLanguages = ["ja-JP", "en-US"]

    do {
        try handler.perform([classifyRequest, textRequest])
    } catch {
        entry["error"] = "vision request failed: \(error)"
    }

    if let observations = classifyRequest.results {
        let labels: [[String: Any]] = observations.prefix(30).map { observation in
            ["id": observation.identifier, "confidence": Double(observation.confidence)]
        }
        entry["classifications"] = labels
    } else {
        entry["classifications"] = []
    }
    entry["text"] = recognizedText
    results.append(entry)
}

emit(results)
