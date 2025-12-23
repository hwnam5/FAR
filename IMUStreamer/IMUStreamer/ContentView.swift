import SwiftUI

struct ContentView: View {
    @StateObject var connectivity = ConnectivityManager.shared
    
    let activities = ["스쿼트", "런지", "팔굽혀펴기", "윗몸일으키기",
                      "플랭크", "팔 벌려 뛰기", "점프 스쿼트", "휴식"]
    
    var body: some View {
        NavigationView {
            List(activities, id: \.self) { activity in
                NavigationLink(
                    destination: DetailView(activity: activity), // 클릭 시 바로 이동
                    label: {
                        Text(activity)
                    }
                )
            }
            .navigationTitle("운동 선택")
            // [위치 변경] 공유 시트를 최상위 뷰에 부착
            .sheet(isPresented: $connectivity.isShowingShareSheet) {
                if let fileURL = connectivity.receivedFileURL {
                    ShareSheet(items: [fileURL])
                }
            }
        }
    }
}

// 상세 화면 (수집 화면)
struct DetailView: View {
    let activity: String
    // 이 뷰에서도 ConnectivityManager에 접근 가능하게 함
    @StateObject var connectivity = ConnectivityManager.shared
    
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "figure.cross.training")
                .font(.system(size: 80))
                .foregroundColor(.blue)
            
            Text("\(activity) 수집 중")
                .font(.largeTitle)
                .bold()
            
            Text("워치에서 [시작] 버튼을 눌러주세요.")
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            Spacer()
        }
        .padding()
        // [핵심 해결책] 화면이 짠! 하고 나타나는 순간 워치로 신호를 쏘아줍니다.
        .onAppear {
            print("아이폰: \(activity) 화면 진입 -> 워치로 신호 전송")
            connectivity.sendActivityChange(activity)
        }
        .onDisappear {
            print("아이폰: 목록으로 돌아감 -> 워치 초기화")
            connectivity.sendActivityChange("선택 안됨")
        }
    }
}