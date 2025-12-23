import SwiftUI
import UIKit

// 아이폰의 기본 공유 창(ActivityViewController)을 SwiftUI에서 쓰기 위한 포장지
struct ShareSheet: UIViewControllerRepresentable {
    var items: [Any] // 공유할 파일들
    
    func makeUIViewController(context: Context) -> UIActivityViewController {
        let controller = UIActivityViewController(activityItems: items, applicationActivities: nil)
        return controller
    }
    
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}