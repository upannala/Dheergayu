pipeline{
    agent{
        label 'master'
    }
    
    stages{
        stage("S3 check"){
            steps{
                script{
                    // git branch: param_map1.find{ it.key == "custom_branch" }.value, credentialsId:'', url: param_map1.find{ it.key == "custom_url" }.value
                    // sh script: "cd custom && chmod +x custom.sh && ./custom.sh ${parameter_1} ${parameter_2}"
                    def cmd= "aws s3 ls s3://videosdhi --recursive | awk \'{print $4}\'"
                    
                    def command = $/ $cmd /$
                    def proc = command.execute()
                    proc.waitFor()              

                    def output = proc.in.text
                    def keys= output.tokenize()

                    for(String key:keys){
                        def uname=key.replaceAll('.mp4','')
                        sh script: "aws s3api get-object --bucket videosdhi --key  ${key} ${key}"
                        sleep(60000)
                        sh script: "python video.py -v ${key} -p shape_predictor_68_face_landmarks.dat -u ${uname}"
                        sh script: "rm -rf *${key}*"
                    }
                }
            }
        }
    }
}
